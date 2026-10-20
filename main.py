from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from itertools import combinations
import logging
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from google import genai
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION (immutable, loaded once)
# =============================================================================


@dataclass(frozen=True)
class HuggingFaceConfig:
    api_key: str
    base_url: str = "https://api-inference.huggingface.co/models"
    models: Dict[str, str] = field(
        default_factory=lambda: {
            "ner": "d4data/biomedical-ner-all",
            "classification": "microsoft/DialoGPT-medium",
            "drug_extraction": "allenai/scibert_scivocab_uncased",
            "clinical_bert": "emilyalsentzer/Bio_ClinicalBERT",
            "medical_qa": "deepset/roberta-base-squad2-covid",
            "drug_interaction": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        }
    )

    @property
    def endpoints(self) -> Dict[str, str]:
        return {k: f"{self.base_url}/{v}" for k, v in self.models.items()}


@lru_cache(maxsize=1)
def get_hf_config() -> HuggingFaceConfig:
    key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
    if not key:
        logger.warning("HUGGINGFACE_API_KEY is not set — HF calls will fail at runtime.")
    return HuggingFaceConfig(api_key=key)


# =============================================================================
# PRE-COMPILED REGEX PATTERNS  (compiled once, reused across requests)
# =============================================================================

_DRUG_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(aspirin|paracetamol|acetaminophen|metformin|warfarin|digoxin|furosemide|lisinopril|"
        r"spironolactone|simvastatin|clopidogrel|rivaroxaban|apixaban|atorvastatin|amlodipine|"
        r"losartan|omeprazole|ibuprofen|naproxen|diclofenac|tramadol|codeine|morphine|amoxicillin|"
        r"ciprofloxacin|azithromycin|prednisolone|hydrocortisone|levothyroxine|insulin|glipizide|"
        r"gliclazide|enalapril|candesartan|bisoprolol|metoprolol|nifedipine|hydrochlorothiazide)"
        r"\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([a-z]{4,}(?:ine|ole|pril|sartan|olol|ide|atin|mycin|cillin|zole|pine|done|sone))"
        r"\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([a-z]{4,})\s+(?:tablets?|capsules?|pills?)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg)\b",
        re.IGNORECASE,
    ),
)

_EXCLUDE_WORDS: FrozenSet[str] = frozenset(
    {
        "take", "taking", "daily", "twice", "once", "with", "without", "after", "before",
        "morning", "evening", "night", "patient", "prescribed", "starting", "consider",
        "also", "tablet", "capsule", "medication", "drug", "dose", "dosage", "treatment",
        "therapy", "management", "pills", "medicine", "continue", "discontinue", "increase",
        "decrease", "reduce", "maintain", "monitor", "check", "follow", "schedule", "routine",
    }
)

_FREQ_MAP: Dict[str, str] = {
    "qd": "once daily",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qhs": "at bedtime",
    "prn": "as needed",
}


def _build_freq_patterns(drug_name: str) -> List[re.Pattern]:
    dn = re.escape(drug_name)
    return [
        re.compile(rf"{dn}.*?\b(once|twice|three times|four times|1x|2x|3x|4x)\s+(?:a\s+)?(?:day|daily)\b", re.IGNORECASE),
        re.compile(rf"{dn}.*?\b(morning|evening|bedtime|night|am|pm)\b", re.IGNORECASE),
        re.compile(rf"{dn}.*?\bevery\s+(\d+)\s+(hours?|hrs?|days?)\b", re.IGNORECASE),
        re.compile(rf"{dn}.*?\b(\d+)\s+times?\s+(?:a\s+|per\s+)?(?:day|daily)\b", re.IGNORECASE),
        re.compile(rf"{dn}.*?\b(q\d+h|qd|bid|tid|qid|qhs|prn)\b", re.IGNORECASE),
    ]


# =============================================================================
# STATIC REFERENCE DATA  (normalized once into efficient lookup structures)
# =============================================================================


def _build_interaction_index() -> Dict[FrozenSet[str], Dict[str, str]]:
    """O(1) bidirectional interaction lookup using frozenset keys."""
    raw: Dict[Tuple[str, str], Dict[str, str]] = {
        ("warfarin", "aspirin"): {
            "severity": "High",
            "description": "Increased bleeding risk - both drugs affect blood clotting mechanisms",
            "recommendation": "Monitor for bleeding signs, consider PPI for gastric protection, frequent INR monitoring",
        },
        ("metformin", "contrast"): {
            "severity": "High",
            "description": "Risk of lactic acidosis and acute kidney injury",
            "recommendation": "Discontinue metformin 48 hours before contrast procedures, resume after kidney function check",
        },
        ("digoxin", "furosemide"): {
            "severity": "Moderate",
            "description": "Furosemide-induced hypokalemia increases digoxin toxicity risk",
            "recommendation": "Monitor potassium and digoxin levels weekly, supplement potassium as needed",
        },
        ("lisinopril", "spironolactone"): {
            "severity": "Moderate",
            "description": "Combined potassium retention may cause dangerous hyperkalemia",
            "recommendation": "Monitor serum potassium every 1-2 weeks, adjust doses accordingly",
        },
        ("warfarin", "simvastatin"): {
            "severity": "Moderate",
            "description": "Simvastatin may potentiate warfarin's anticoagulant effects",
            "recommendation": "Monitor INR more frequently when initiating or adjusting simvastatin",
        },
        ("metformin", "furosemide"): {
            "severity": "Low",
            "description": "Furosemide may affect renal function and metformin clearance",
            "recommendation": "Monitor renal function and glucose control regularly",
        },
        ("aspirin", "ibuprofen"): {
            "severity": "Moderate",
            "description": "Increased gastrointestinal bleeding risk with dual NSAIDs",
            "recommendation": "Avoid combination, use gastroprotective agents if necessary",
        },
        ("warfarin", "antibiotics"): {
            "severity": "High",
            "description": "Many antibiotics can significantly alter warfarin metabolism",
            "recommendation": "Increase INR monitoring frequency during antibiotic therapy",
        },
    }
    return {frozenset(pair): data for pair, data in raw.items()}


DRUG_INTERACTION_INDEX: Dict[FrozenSet[str], Dict[str, str]] = _build_interaction_index()

AGE_DOSING: Dict[str, Dict[str, Dict[str, Any]]] = {
    "aspirin": {
        "child": {"max_age": 17, "dosage": "CONTRAINDICATED - Risk of Reye's syndrome"},
        "adult": {"max_age": 64, "dosage": "75-325mg daily for cardioprotection"},
        "elderly": {"max_age": 120, "dosage": "75-100mg daily (reduced bleeding risk)"},
    },
    "paracetamol": {
        "child": {"max_age": 17, "dosage": "10-15mg/kg every 4-6 hours (max 60mg/kg/day)"},
        "adult": {"max_age": 64, "dosage": "500-1000mg every 4-6 hours (max 4g/day)"},
        "elderly": {"max_age": 120, "dosage": "500mg every 6-8 hours (hepato-renal protection)"},
    },
    "metformin": {
        "adult": {"max_age": 64, "dosage": "500-1000mg twice daily with meals"},
        "elderly": {"max_age": 120, "dosage": "500mg twice daily (monitor eGFR closely)"},
    },
    "warfarin": {
        "adult": {"max_age": 64, "dosage": "2.5-10mg daily (target INR 2.0-3.0)"},
        "elderly": {"max_age": 120, "dosage": "1.25-5mg daily (start low, go slow approach)"},
    },
    "digoxin": {
        "adult": {"max_age": 64, "dosage": "125-250mcg daily (check levels)"},
        "elderly": {"max_age": 120, "dosage": "62.5-125mcg daily (reduced clearance in elderly)"},
    },
}

ALTERNATIVES: Dict[str, List[Dict[str, str]]] = {
    "aspirin": [
        {"name": "clopidogrel", "reason": "Reduced GI bleeding risk with similar antiplatelet efficacy", "dosage": "75mg daily"},
        {"name": "rivaroxaban", "reason": "Direct oral anticoagulant with fewer food/drug interactions", "dosage": "20mg daily"},
    ],
    "warfarin": [
        {"name": "apixaban", "reason": "DOAC with lower bleeding risk and no routine monitoring", "dosage": "5mg twice daily"},
        {"name": "rivaroxaban", "reason": "Once-daily DOAC with predictable pharmacokinetics", "dosage": "20mg daily"},
    ],
    "metformin": [
        {"name": "sitagliptin", "reason": "DPP-4 inhibitor safe in renal impairment", "dosage": "100mg daily"},
        {"name": "empagliflozin", "reason": "SGLT2 inhibitor with cardiovascular and renal benefits", "dosage": "10-25mg daily"},
    ],
}


def _build_allergy_index() -> Dict[str, FrozenSet[str]]:
    """Normalized allergy class -> members lookup for O(1) membership tests."""
    raw: Dict[str, List[str]] = {
        "aspirin": ["salicylates", "nsaid", "aspirin", "salicylic acid"],
        "penicillin": ["penicillin", "beta-lactam", "amoxicillin", "ampicillin", "methicillin"],
        "sulfa": ["sulfonamides", "sulfur", "sulfa", "sulfamethoxazole", "sulfasalazine"],
        "warfarin": ["warfarin", "coumarin", "anticoagulant"],
        "simvastatin": ["statin", "simvastatin", "lovastatin", "hmg-coa reductase inhibitor"],
    }
    return {k.lower(): frozenset(v) for k, v in raw.items()}


ALLERGY_INDEX: Dict[str, FrozenSet[str]] = _build_allergy_index()


# =============================================================================
# DATA MODELS
# =============================================================================


class DrugInfo(BaseModel):
    name: str = Field(..., min_length=1, description="Drug name")
    dosage: Optional[str] = Field(None, description="Drug dosage with units")
    frequency: Optional[str] = Field(None, description="Dosing frequency")
    ai_confidence: Optional[float] = Field(None, description="AI extraction confidence score")


class PatientInfo(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    weight: Optional[float] = Field(None, ge=0, le=500, description="Patient weight in kg")
    conditions: List[str] = Field(default_factory=list, description="Medical conditions")
    allergies: List[str] = Field(default_factory=list, description="Known drug allergies")


class PrescriptionRequest(BaseModel):
    drugs: List[DrugInfo] = Field(default_factory=list)
    patient: PatientInfo
    medical_text: Optional[str] = None
    use_ai_analysis: bool = True


class AIInsight(BaseModel):
    source: str
    insight: str
    confidence: float
    type: str


class EnhancedAnalysisResponse(BaseModel):
    interactions: List[Dict[str, Any]] = Field(default_factory=list)
    dosage_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_drugs: List[DrugInfo] = Field(default_factory=list)
    allergy_warnings: List[Dict[str, Any]] = Field(default_factory=list)
    safety_score: float = Field(..., ge=0.0, le=100.0)
    warnings: List[str] = Field(default_factory=list)
    ai_insights: List[str] = Field(default_factory=list)
    huggingface_analysis: Dict[str, Any] = Field(default_factory=dict)
    gemini_analysis: Dict[str, Any] = Field(default_factory=dict)
    enhanced_ai_insights: List[AIInsight] = Field(default_factory=list)


# =============================================================================
# AI SERVICE LAYER  (connection-pooled httpx clients, DI-ready)
# =============================================================================


class HuggingFaceService:
    """HuggingFace integration using a shared, connection-pooled httpx client."""

    __slots__ = ("_config", "_client", "_headers")

    def __init__(self, client: httpx.AsyncClient, config: HuggingFaceConfig) -> None:
        self._config = config
        self._client = client
        self._headers = {"Authorization": f"Bearer {config.api_key}"}

    # ------------------------------------------------------------------ NER
    async def extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        url = self._config.endpoints["ner"]
        try:
            resp = await self._client.post(url, headers=self._headers, json={"inputs": text})
            if resp.status_code == 200:
                return self._process_ner(resp.json())
            logger.warning("HF NER HTTP %s: %s", resp.status_code, resp.text[:200])
        except httpx.HTTPError as exc:
            logger.error("HF NER call failed: %s", exc)
        return []

    @staticmethod
    def _process_ner(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        keep_labels = frozenset({"DRUG", "MEDICATION", "CHEMICAL"})
        return [
            {
                "text": e.get("word", ""),
                "label": e.get("entity_group", ""),
                "confidence": e.get("score", 0.0),
                "start": e.get("start", 0),
                "end": e.get("end", 0),
            }
            for e in entities
            if e.get("entity_group") in keep_labels
        ]

    # ------------------------------------------------- Interaction analysis
    async def analyze_drug_interactions(self, drug_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Fan-out concurrent requests — each pair is an independent API call,
        but they share a single pooled httpx client with keep-alive connections."""
        if not drug_pairs:
            return []

        url = self._config.endpoints["drug_interaction"]

        async def _one(d1: str, d2: str) -> Optional[Dict[str, Any]]:
            try:
                resp = await self._client.post(
                    url,
                    headers=self._headers,
                    json={"inputs": f"Drug interaction between {d1} and {d2}"},
                )
                if resp.status_code == 200:
                    return {
                        "drug1": d1,
                        "drug2": d2,
                        "analysis": resp.json(),
                        "source": "huggingface",
                    }
            except httpx.HTTPError as exc:
                logger.error("HF interaction (%s/%s) failed: %s", d1, d2, exc)
            return None

        results = await asyncio.gather(*(_one(a, b) for a, b in drug_pairs), return_exceptions=False)
        return [r for r in results if r is not None]

    # -------------------------------------------------- Medical insights
    async def get_medical_insights(self, prescription_text: str) -> Dict[str, Any]:
        url = self._config.endpoints["clinical_bert"]
        try:
            resp = await self._client.post(
                url,
                headers=self._headers,
                json={
                    "inputs": f"Analyze this prescription: {prescription_text}",
                    "parameters": {"max_length": 512},
                },
            )
            if resp.status_code == 200:
                return {
                    "insights": resp.json(),
                    "source": "huggingface_clinical_bert",
                    "status": "success",
                }
        except httpx.HTTPError as exc:
            logger.error("HF medical insights failed: %s", exc)
        return {"insights": None, "source": "huggingface", "status": "failed"}


class GeminiService:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.client = None
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set — Gemini calls are disabled.")
            return

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception:
            logger.exception("Failed to initialize Gemini client.")

    async def _generate_content(self, prompt: str, analysis_type: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "analysis": None,
                "source": "gemini",
                "status": "not_configured",
            }

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = response.text
            if not text:
                logger.warning("Gemini returned an empty %s response.", analysis_type)
                return {
                    "analysis": None,
                    "source": "gemini",
                    "status": "failed",
                }
            return {
                "analysis": text,
                "source": "gemini",
                "status": "success",
            }
        except Exception:
            logger.exception("Gemini %s analysis failed.", analysis_type)
            return {
                "analysis": None,
                "source": "gemini",
                "status": "failed",
            }

    async def get_dosage_recommendations(
        self, drugs: List[str], patient: Dict[str, Any]
    ) -> Dict[str, Any]:
        patient_context = {
            "age": patient.get("age"),
            "weight": patient.get("weight"),
            "conditions": patient.get("conditions", []),
            "allergies": patient.get("allergies", []),
        }
        prompt = (
            "Provide general, non-prescriptive dosage reference considerations for "
            f"these medications: {', '.join(drugs)}. Patient context: "
            f"{patient_context}. Do not prescribe or recommend changing a dose. "
            "Flag missing information and explain that a licensed clinician must "
            "verify any dosage against current authoritative references."
        )
        return await self._generate_content(prompt, "dosage")

    async def check_safety_guidelines(
        self, drugs: List[str], patient: Dict[str, Any]
    ) -> Dict[str, Any]:
        patient_context = {
            "age": patient.get("age"),
            "conditions": patient.get("conditions", []),
            "allergies": patient.get("allergies", []),
        }
        prompt = (
            "Review these medications for general safety considerations, "
            "contraindications, and potential interactions: "
            f"{', '.join(drugs)}. Patient context: {patient_context}. "
            "Identify uncertainty and advise verification by a licensed clinician; "
            "do not present this as a diagnosis or a substitute for professional care."
        )
        return await self._generate_content(prompt, "safety")


# =============================================================================
# LIFESPAN + DEPENDENCY INJECTION
# =============================================================================


@dataclass
class ServiceContainer:
    hf: HuggingFaceService
    gemini_service: GeminiService
    http: httpx.AsyncClient


def _build_http_client() -> httpx.AsyncClient:
    """Build the shared pooled httpx client; gracefully degrade to HTTP/1.1 if h2 missing."""
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=30.0)
    timeout = httpx.Timeout(60.0, connect=10.0)
    try:
        return httpx.AsyncClient(limits=limits, timeout=timeout, http2=True)
    except ImportError:
        logger.warning("h2 package not installed — falling back to HTTP/1.1 (install httpx[http2] for HTTP/2)")
        return httpx.AsyncClient(limits=limits, timeout=timeout, http2=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the shared Hugging Face HTTP client for the process lifetime."""
    client = _build_http_client()
    try:
        hf = HuggingFaceService(client, get_hf_config())
        gemini_service = GeminiService()
        app.state.services = ServiceContainer(
            hf=hf, gemini_service=gemini_service, http=client
        )
        logger.info("🚀 Enhanced AI Prescription API v3.0.0 starting")
        yield
    finally:
        await client.aclose()
        logger.info("🔄 Shutting down — httpx pool drained")


def get_services(request: Request) -> ServiceContainer:
    return request.app.state.services


app = FastAPI(
    title="AI Medical Prescription Verification API",
    version="3.0.0",
    description="Advanced AI-powered drug interaction and safety analysis system",
    lifespan=lifespan,
)

_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)


# =============================================================================
# NLP / DRUG EXTRACTION  (pure, stateless, testable functions)
# =============================================================================


def extract_dosage_from_context(text: str, drug_name: str) -> Optional[str]:
    pattern = rf"{re.escape(drug_name)}\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)"
    m = re.search(pattern, text, re.IGNORECASE)
    return f"{m.group(1)}{m.group(2)}" if m else None


def extract_frequency(text: str, drug_name: str) -> Optional[str]:
    for pattern in _build_freq_patterns(drug_name):
        m = pattern.search(text)
        if not m:
            continue
        raw = m.group(1)
        return _FREQ_MAP.get(raw.lower(), raw)
    return None


def extract_drugs_from_text(text: str) -> List[DrugInfo]:
    """Rule-based extraction. Returns unique drugs via O(1) set membership."""
    if not text or len(text.strip()) < 3:
        return []

    normalized = text.lower().replace("mg.", "mg").replace("mcg.", "mcg")
    seen: set[str] = set()
    out: List[DrugInfo] = []

    for pattern in _DRUG_PATTERNS:
        for m in pattern.finditer(normalized):
            name = m.group(1).strip()
            if name in _EXCLUDE_WORDS or len(name) < 3:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)

            try:
                amt, unit = m.group(2), m.group(3)
                dosage = f"{amt}{unit}"
            except (IndexError, AttributeError):
                dosage = None

            out.append(
                DrugInfo(
                    name=name,
                    dosage=dosage,
                    frequency=extract_frequency(normalized, name),
                )
            )
    return out


async def enhanced_drug_extraction(text: str, use_ai: bool, services: ServiceContainer) -> List[DrugInfo]:
    """Hybrid rule + AI extraction. Deduplicates via normalized name set."""
    if not text or len(text.strip()) < 3:
        return []

    rule_based = extract_drugs_from_text(text)
    ai_found: List[DrugInfo] = []

    if use_ai:
        try:
            entities = await services.hf.extract_medical_entities(text)
            for e in entities:
                if e["label"] not in ("DRUG", "MEDICATION", "CHEMICAL"):
                    continue
                name = e["text"].strip()
                if len(name) <= 2:
                    continue
                ai_found.append(
                    DrugInfo(
                        name=name,
                        dosage=extract_dosage_from_context(text, name),
                        frequency=extract_frequency(text.lower(), name.lower()),
                        ai_confidence=e["confidence"],
                    )
                )
        except Exception as exc:  # noqa: BLE001 — AI is best-effort; never break the request
            logger.error("HF extraction failed, falling back to rules only: %s", exc)

    # Union-deduplicate in a single pass (no nested loops).
    seen: set[str] = set()
    unique: List[DrugInfo] = []
    for d in (*rule_based, *ai_found):
        key = d.name.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


# =============================================================================
# CLINICAL ANALYSIS ENGINE  (pure functions)
# =============================================================================


def check_interactions(drugs: List[DrugInfo]) -> List[Dict[str, Any]]:
    """O(n²) pair generation is inherent; each lookup is O(1) via frozenset index."""
    if len(drugs) < 2:
        return []
    out: List[Dict[str, Any]] = []
    for a, b in combinations(drugs, 2):
        key = frozenset((a.name.lower().strip(), b.name.lower().strip()))
        data = DRUG_INTERACTION_INDEX.get(key)
        if data:
            out.append(
                {
                    "drug1": a.name.title(),
                    "drug2": b.name.title(),
                    "severity": data["severity"],
                    "description": data["description"],
                    "recommendation": data["recommendation"],
                }
            )
    return out


def get_dosage_recommendations(drugs: List[DrugInfo], patient_age: int) -> List[Dict[str, Any]]:
    if patient_age < 18:
        bucket = "child"
    elif patient_age <= 64:
        bucket = "adult"
    else:
        bucket = "elderly"

    out: List[Dict[str, Any]] = []
    for d in drugs:
        info = AGE_DOSING.get(d.name.lower().strip())
        if not info:
            continue
        dose = info.get(bucket) or info.get("adult")
        if not dose:
            continue
        out.append(
            {
                "drug_name": d.name.title(),
                "current_dosage": d.dosage or "Not specified",
                "recommended_dosage": dose["dosage"],
                "notes": f"Age group: {bucket} ({patient_age} years)",
            }
        )
    return out


def check_allergies(drugs: List[DrugInfo], patient_allergies: Iterable[str]) -> List[Dict[str, Any]]:
    """O(drugs × |allergen classes|) worst-case; each step uses set ops.
    ALLERGY_INDEX is keyed by allergen *class name* (e.g. "penicillin"),
    so we match in two passes: direct-name hit + class-membership scan."""
    if not patient_allergies:
        return []

    allergy_tokens: set[str] = set()
    for raw in patient_allergies:
        token = raw.lower().strip()
        if not token:
            continue
        allergy_tokens.add(token)
        allergy_tokens.update(token.split())

    out: List[Dict[str, Any]] = []
    for d in drugs:
        key = d.name.lower().strip()
        match_allergen: Optional[str] = None

        # 1) Direct-name allergy hit (drug itself listed in patient allergies)
        if key in allergy_tokens:
            match_allergen = key

        # 2) Class-membership: drug is a member of an allergen class the patient has.
        #    Iterate ALLERGY_INDEX items: (allergen_class_name, members_frozenset).
        if match_allergen is None:
            for allergen_class, members in ALLERGY_INDEX.items():
                if key not in members:
                    continue
                overlap = ({allergen_class} | members) & allergy_tokens
                if overlap:
                    match_allergen = next(iter(overlap))
                    break

        if match_allergen is None:
            continue

        out.append(
            {
                "drug_name": d.name.title(),
                "allergen": match_allergen,
                "severity": "High",
                "recommendation": f"CONTRAINDICATED: {d.name.title()} due to {match_allergen} allergy",
            }
        )
    return out


def suggest_alternatives(
    drugs: List[DrugInfo],
    interactions: List[Dict[str, Any]],
    allergy_warnings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    problematic: set[str] = set()
    for it in interactions:
        if it["severity"] == "High":
            problematic.add(it["drug1"].lower())
            problematic.add(it["drug2"].lower())
    for w in allergy_warnings:
        problematic.add(w["drug_name"].lower())

    out: List[Dict[str, Any]] = []
    for d in drugs:
        key = d.name.lower().strip()
        if key not in problematic:
            continue
        for alt in ALTERNATIVES.get(key, ()):
            out.append(
                {
                    "original_drug": d.name.title(),
                    "alternative": alt["name"].title(),
                    "reason": alt["reason"],
                    "dosage": alt["dosage"],
                }
            )
    return out


def calculate_safety_score(
    interactions: List[Dict[str, Any]],
    allergy_warnings: List[Dict[str, Any]],
    patient_age: int,
    drug_count: int,
    _conditions: Iterable[str],
) -> float:
    score = 100.0
    score -= len(allergy_warnings) * 45.0

    severity_scores = {"High": 30.0, "Moderate": 18.0, "Low": 8.0}
    for it in interactions:
        score -= severity_scores.get(it["severity"], 5.0)

    if patient_age < 18:
        score -= 15.0
    elif patient_age > 80:
        score -= 20.0
    elif patient_age > 65:
        score -= 10.0

    if drug_count >= 6:
        score -= 25.0
    elif drug_count >= 4:
        score -= 15.0

    return max(score, 0.0)


def generate_ai_insights(drugs: List[DrugInfo], interactions: List[Dict[str, Any]], patient_age: int, drug_count: int) -> List[str]:
    out: List[str] = []
    high_risk = sum(1 for i in interactions if i["severity"] == "High")
    if high_risk:
        out.append(f"🤖 AI Critical Alert: {high_risk} high-risk interactions detected")
    if patient_age > 75:
        out.append("🧓 AI Senior Care: Age-adjusted dosing recommended")
    if drug_count >= 5:
        out.append("💊 AI Polypharmacy Alert: Complex regimen requires monitoring")
    return out


def generate_warnings(
    interactions: List[Dict[str, Any]],
    allergy_warnings: List[Dict[str, Any]],
    patient_age: int,
    drug_count: int,
    _conditions: Iterable[str],
) -> List[str]:
    out: List[str] = []
    if allergy_warnings:
        out.append(f"🚨 CRITICAL: {len(allergy_warnings)} allergy contraindications!")
    high = [i for i in interactions if i["severity"] == "High"]
    if high:
        out.append(f"⚠️ URGENT: {len(high)} high-risk interactions")
    if patient_age > 80:
        out.append("👴 GERIATRIC ALERT: High-risk age group")
    if drug_count >= 7:
        out.append("💊 HIGH POLYPHARMACY: Complex regimen")
    return out


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/")
async def root() -> Dict[str, Any]:
    hf_cfg = get_hf_config()
    gemini_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
    return {
        "message": "🏥 AI Medical Prescription Verification System",
        "status": "🟢 Online and Ready",
        "version": "3.0.0",
        "ai_services": {
            "huggingface": {
                "status": "🟢 Connected" if hf_cfg.api_key else "🟡 Not configured",
                "models": list(hf_cfg.models.keys()),
            },
            "gemini": {
                "status": "🟢 Configured" if gemini_configured else "🟡 Not configured",
                "model": "gemini-2.5-flash",
            },
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    hf_cfg = get_hf_config()
    gemini_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "ai_services": {
            "huggingface": {
                "status": "connected" if hf_cfg.api_key else "not_configured",
                "models_available": len(hf_cfg.models),
            },
            "gemini": {
                "status": "configured" if gemini_configured else "not_configured",
                "model": "gemini-2.5-flash",
            },
        },
    }


@app.post("/analyze", response_model=EnhancedAnalysisResponse)
async def analyze_prescription(
    request: PrescriptionRequest,
    services: ServiceContainer = Depends(get_services),
):
    logger.info("Starting analysis for patient age %s", request.patient.age)

    # 1) Hybrid NLP extraction
    extracted_drugs: List[DrugInfo] = []
    if request.medical_text and request.medical_text.strip():
        extracted_drugs = await enhanced_drug_extraction(
            request.medical_text, request.use_ai_analysis, services
        )
        logger.info("AI extraction found %d drugs", len(extracted_drugs))

    # 2) Union-deduplicate user-provided + extracted (single-pass, O(n) via set)
    seen: set[str] = set()
    unique_drugs: List[DrugInfo] = []
    for d in (*request.drugs, *extracted_drugs):
        key = d.name.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_drugs.append(d)

    if not unique_drugs:
        raise HTTPException(status_code=400, detail="No valid drugs found")

    # 3) Fan out AI calls concurrently.
    ai_tasks: List[asyncio.Future] = []
    if request.use_ai_analysis:
        names = [d.name for d in unique_drugs]
        pairs = list(combinations(names, 2))
        if pairs:
            ai_tasks.append(services.hf.analyze_drug_interactions(pairs))
        ai_tasks.append(services.hf.get_medical_insights(request.medical_text or ""))

        patient_data = {
            "age": request.patient.age,
            "weight": request.patient.weight,
            "conditions": request.patient.conditions,
            "allergies": request.patient.allergies,
        }
        ai_tasks.append(
            services.gemini_service.check_safety_guidelines(names, patient_data)
        )
        ai_tasks.append(
            services.gemini_service.get_dosage_recommendations(names, patient_data)
        )

    # 4) Classical analysis runs concurrently with the network I/O above by
    #    scheduling it as a CPU-bound task in the threadpool.
    async def _classical():
        return (
            check_interactions(unique_drugs),
            get_dosage_recommendations(unique_drugs, request.patient.age),
            check_allergies(unique_drugs, request.patient.allergies),
        )

    classical_task = asyncio.create_task(_classical())

    ai_results: List[Any] = []
    if ai_tasks:
        # Exceptions from individual AI calls are handled inside services;
        # using return_exceptions here is a safety net so one bad call never
        # poisons the full gather.
        raw = await asyncio.gather(*ai_tasks, return_exceptions=True)
        ai_results = [r if not isinstance(r, BaseException) else None for r in raw]

    interactions, dosage_recs, allergy_warnings = await classical_task
    alternatives = suggest_alternatives(unique_drugs, interactions, allergy_warnings)
    safety_score = calculate_safety_score(
        interactions, allergy_warnings, request.patient.age,
        len(unique_drugs), request.patient.conditions,
    )
    ai_insights = generate_ai_insights(unique_drugs, interactions, request.patient.age, len(unique_drugs))
    warnings = generate_warnings(
        interactions, allergy_warnings, request.patient.age,
        len(unique_drugs), request.patient.conditions,
    )

    # 5) Assemble AI enrichments (tuple unpack is order-preserving w.r.t. ai_tasks)
    hf_analysis: Dict[str, Any] = {}
    gemini_analysis: Dict[str, Any] = {}
    enhanced: List[AIInsight] = []

    if request.use_ai_analysis and ai_results:
        idx = 0
        if pairs and idx < len(ai_results) and ai_results[idx] is not None:
            hf_analysis["drug_interactions"] = ai_results[idx]
        idx += 1 if pairs else 0

        if idx < len(ai_results) and ai_results[idx] is not None:
            hf_ins = ai_results[idx]
            hf_analysis["medical_insights"] = hf_ins
            if hf_ins.get("status") == "success":
                enhanced.append(
                    AIInsight(
                        source="HuggingFace Clinical BERT",
                        insight="Advanced NLP analysis completed with clinical text understanding",
                        confidence=0.85,
                        type="clinical_analysis",
                    )
                )
        idx += 1

        if idx < len(ai_results) and ai_results[idx] is not None:
            gemini_safety = ai_results[idx]
            gemini_analysis["safety_analysis"] = gemini_safety
            if gemini_safety.get("status") == "success":
                enhanced.append(
                    AIInsight(
                        source="Google Gemini",
                        insight="Prescription safety review completed",
                        confidence=0.88,
                        type="safety_analysis",
                    )
                )
        idx += 1

        if idx < len(ai_results) and ai_results[idx] is not None:
            gemini_dosage = ai_results[idx]
            gemini_analysis["dosage_recommendations"] = gemini_dosage
            if gemini_dosage.get("status") == "success":
                enhanced.append(
                    AIInsight(
                        source="Google Gemini",
                        insight="General dosage reference review completed",
                        confidence=0.82,
                        type="recommendations",
                    )
                )

    logger.info("Analysis complete — safety=%.1f interactions=%d allergies=%d",
                safety_score, len(interactions), len(allergy_warnings))

    return EnhancedAnalysisResponse(
        interactions=interactions,
        dosage_recommendations=dosage_recs,
        alternatives=alternatives,
        extracted_drugs=unique_drugs,
        allergy_warnings=allergy_warnings,
        safety_score=safety_score,
        warnings=warnings,
        ai_insights=ai_insights,
        huggingface_analysis=hf_analysis,
        gemini_analysis=gemini_analysis,
        enhanced_ai_insights=enhanced,
    )


@app.post("/ai-insights")
async def get_ai_insights(
    payload: Dict[str, Any],
    services: ServiceContainer = Depends(get_services),
) -> Dict[str, Any]:
    medical_text = payload.get("medical_text", "")
    patient_data = payload.get("patient_data", {})
    if not medical_text and not patient_data:
        raise HTTPException(status_code=400, detail="Medical text or patient data required")

    tasks: Dict[str, asyncio.Future] = {}
    if medical_text:
        tasks["hf_entities"] = asyncio.create_task(services.hf.extract_medical_entities(medical_text))
        tasks["hf_insights"] = asyncio.create_task(services.hf.get_medical_insights(medical_text))
    if patient_data:
        drugs = patient_data.get(
            "drugs",
            patient_data.get("medications", patient_data.get("current_medications", [])),
        )
        if isinstance(drugs, str):
            drugs = [drugs]
        if not isinstance(drugs, list):
            drugs = []
        drugs = [
            drug.strip()
            if isinstance(drug, str)
            else drug["name"].strip()
            for drug in drugs
            if (isinstance(drug, str) and drug.strip())
            or (
                isinstance(drug, dict)
                and isinstance(drug.get("name"), str)
                and drug["name"].strip()
            )
        ]
        tasks["gemini_safety"] = asyncio.create_task(
            services.gemini_service.check_safety_guidelines(drugs, patient_data)
        )
        tasks["gemini_dosage"] = asyncio.create_task(
            services.gemini_service.get_dosage_recommendations(drugs, patient_data)
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    resolved = dict(zip(tasks.keys(), (r if not isinstance(r, BaseException) else None for r in results)))

    insights: Dict[str, Any] = {}
    if medical_text:
        insights["huggingface"] = {
            "entities": resolved.get("hf_entities") or [],
            "medical_insights": resolved.get("hf_insights") or {},
            "timestamp": datetime.now().isoformat(),
        }
    if patient_data:
        insights["gemini"] = {
            "safety_analysis": resolved.get("gemini_safety") or {},
            "dosage_recommendations": resolved.get("gemini_dosage") or {},
            "timestamp": datetime.now().isoformat(),
        }

    return {
        "status": "success",
        "insights": insights,
        "ai_services_used": list(insights.keys()),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True,
    )