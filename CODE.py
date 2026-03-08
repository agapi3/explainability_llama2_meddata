
import os
import re
import json
import pickle
import hashlib
import logging
import warnings
import time
import signal
import threading
import ast
import unicodedata
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional
from functools import lru_cache
from difflib import SequenceMatcher
from collections import Counter, OrderedDict
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain_ollama import OllamaLLM


#load dataset
ds = load_dataset("Malikeh1375/medical-question-answering-datasets", "chatdoctor_healthcaremagic") 
ds_healthcaremagic = ds["train"]

# JSON
with open("healthcaremagic_dataset.json", "w", encoding="utf-8") as f:
    json.dump(ds_healthcaremagic.to_dict(), f, ensure_ascii=False, indent=4)


with open("healthcaremagic_dataset.json", "r", encoding="utf-8") as f:
    ds_healthcaremagic = json.load(f)

ds_healthcaremagic = ds["train"]
print(ds_healthcaremagic[:5])

#check 
print(len(ds_healthcaremagic))
print("NaN values per column (before cleaning):")
print(pd.DataFrame(ds_healthcaremagic).isna().sum())
print("Total duplicates (before cleaning):", pd.DataFrame(ds_healthcaremagic).duplicated(subset=["instruction","input","output"]).sum())


#data preprocessing
def clean_text(text):
    if text is None:
        return ""

    #ensure string&normalize unicode (keeps text stable across sources)
    text = unicodedata.normalize("NFKC", str(text))

    #trim&lowercase
    text = text.strip().lower()

    #normalize fancy dashes to simple hyphen
    text = re.sub(r"[–—−]", "-", text)

    # IMPORTANT: prevent token concatenation on slashes
    # "leg/surgery" -> "leg surgery", "rattly/raspy" -> "rattly raspy"
    text = re.sub(r"\s*/\s*", " ", text)

    #keep clinically meaningful punctuation and separators
    #keep: . , ! ? ' - ( ) : ;
    # (Keeping '-' avoids 5-7 -> 57)
    text = re.sub(r"[^\w\s\.\,\!\?\-\'\(\)\:\;]", " ", text)

    #collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prettify_text(text):
    if not text:
        return ""
    text = text.strip()
    text = text[0].upper() + text[1:]  #capitalize first letter
    if text[-1] not in ".!?":          #ensure proper punctuation
        text += "."
    return text

#low-info/stub output filter
def is_low_info_output(output_text, min_words=25):
    """
    Optional quality gate for reference answers.
    Use ONLY if you decide to exclude low-quality ground-truth outputs.
    """
    if not output_text:
        return True

    t = output_text.strip().lower()
    words = t.split()

    #very short answers are usually not useful as "ground truth"
    if len(words) < min_words:
        return True

    #obvious truncation marker in many scraped corpora
    if t.endswith("..."):
        return True

    
    boiler = ["dear friend", "welcome to chat doctor", "i am chat doctor"]
    if any(b in t for b in boiler) and len(words) < (min_words + 10):
        return True

    return False

#clean and prettify dataset
cleaned_pretty_healthcaremagic = []


APPLY_STUB_FILTER = False

dropped_empty = 0
dropped_stub = 0

for example in ds_healthcaremagic:
    instruction = example['instruction'][0] if isinstance(example['instruction'], list) else example['instruction']
    input_text = example['input']
    response = example['output']

    cleaned_instruction = clean_text(instruction)
    cleaned_input = clean_text(input_text)
    cleaned_response = clean_text(response)

    #skip empty fields
    if not cleaned_instruction or not cleaned_input or not cleaned_response:
        dropped_empty += 1
        continue

    #filter low-information / truncated reference outputs
    if APPLY_STUB_FILTER and is_low_info_output(cleaned_response):
        dropped_stub += 1
        continue

    pretty_instruction = prettify_text(cleaned_instruction)
    pretty_input = prettify_text(cleaned_input)
    pretty_response = prettify_text(cleaned_response)

    cleaned_pretty_healthcaremagic.append({
        "instruction": pretty_instruction,
        "input": pretty_input,
        "output": pretty_response
    })

#remove duplicates
df = pd.DataFrame(cleaned_pretty_healthcaremagic)
before = len(df)
df = df.drop_duplicates(subset=["instruction", "input", "output"])
after = len(df)

cleaned_pretty_healthcaremagic = df.to_dict(orient="records")

with open("cleaned_healthcaremagic.pkl", "wb") as f:
    pickle.dump(cleaned_pretty_healthcaremagic, f)

print("Saved cleaned_healthcaremagic.pkl:", len(cleaned_pretty_healthcaremagic), "samples")
print("Dropped (empty fields):", dropped_empty)
print("Dropped (stub/low-info outputs):", dropped_stub)
print("Duplicates removed:", before - after)
print("NaN values per column (after cleaning):")
print(df.isna().sum())
print("Total duplicates (after cleaning):", df.duplicated(subset=["instruction","input","output"]).sum())


# ---------------------------
# PRINT/LOG CONTROLS
# ---------------------------
LOG_LEVEL = "ERROR"            
PRINT_EVERY = 1                
PRINT_CF_TEXT = True           
PRINT_MAX_CF_CHARS = 2000      
PRINT_TRUNC_INPUT_CHARS = 800
PRINT_TRUNC_OUTPUT_CHARS = 800


PRINT_TIMERS = False           
PRINT_EVAL_START = False       
PRINT_LOAD_LINES = True        


#SPEED/STABILITY (Mac)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NLTK_DATA", "/Users/agapikyrimi/nltk_data")

warnings.filterwarnings("ignore")


#thread tuning
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except Exception:
    pass


#OPTIONAL
SEED = 1234
try:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except Exception:
    pass


#CONFIG
MODE_TO_RUN = "BOTH"        
MAX_RETRIES = 1               #retries for CF generation
MAX_ITEMS = 100               

CACHE_FLUSH_EVERY = 100
INFLUENTIAL_TOPK_MAX = 2  

#metrics
EVAL_BERT = True
EVAL_NLI = True
EVAL_METEOR = True
EVAL_ROUGE = True

BERT_MODEL_PATH = "/Users/agapikyrimi/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
LLM_MODEL_NAME = "llama2"
LLM_TEMPERATURE = 0.0

NLI_MODEL_NAME = "facebook/bart-large-mnli"
NLI_MAX_LEN = 512

TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 1
TFIDF_MAX_FEATURES = 100000


LLM_TIMEOUT_SEC = 90

EMB_CACHE_MAX = 10_000
TFIDF_VEC_CACHE_MAX = 2_000
NLI_LRU_MAX = 10_000
LLM_LRU_MAX = 10_000


OLLAMA_KEEP_ALIVE: Optional[str] = "30m"


ERROR_ANALYSIS_ONLY = False

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()


#versions/repro metadata
def get_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        versions["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        versions["torch"] = "unknown"
    try:
        import transformers as _tf
        versions["transformers"] = getattr(_tf, "__version__", "unknown")
    except Exception:
        versions["transformers"] = "unknown"
    try:
        import sentence_transformers as _st
        versions["sentence_transformers"] = getattr(_st, "__version__", "unknown")
    except Exception:
        versions["sentence_transformers"] = "unknown"
    return versions


#files/logging
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath(os.getcwd())
os.makedirs(SCRIPT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(
    SCRIPT_DIR,
    f"stored_results_expl_{MODE_TO_RUN}_bert{int(EVAL_BERT)}_nli{int(EVAL_NLI)}_met{int(EVAL_METEOR)}_rouge{int(EVAL_ROUGE)}.pkl"
)
RUNLOG_FILE = os.path.join(SCRIPT_DIR, f"runlog_expl_{MODE_TO_RUN}.txt")

handlers = [logging.StreamHandler()]
try:
    handlers.append(logging.FileHandler(RUNLOG_FILE, mode="a", encoding="utf-8"))
except Exception:
    handlers = [logging.StreamHandler()]

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.ERROR),
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=handlers
)

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            stored_results = pickle.load(f)
    except Exception:
        stored_results = {}
else:
    stored_results = {}


#RSS memory logging
def rss_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None

def _shorten(s: str, n: Optional[int]) -> str:
    s = str(s or "")
    if n is None:
        return s
    return s if len(s) <= n else s[:n] + "...[TRUNCATED]"

def _print_sample(res: Dict[str, Any], idx1: int, total: int, h: str):
    print("\n" + "=" * 100)
    print(f"SAMPLE {idx1}/{total} | hash={h}")
    print("-" * 100)
    print("ORIGINAL INPUT:\n", _shorten(res.get("original_input", ""), PRINT_TRUNC_INPUT_CHARS))
    print("\nORIGINAL OUTPUT:\n", _shorten(res.get("original_output", ""), PRINT_TRUNC_OUTPUT_CHARS))

    print("\nINFLUENTIAL WORDS:", res.get("term_candidates_influential", []))
    print("TF-IDF RAW:", res.get("tfidf_weights_raw", {}))
    print("TF-IDF NORM:", res.get("tfidf_weights_norm", {}))

    cfr = res.get("counterfactuals_robust", []) or []
    cfs = res.get("counterfactuals_strict", []) or []

    print(f"\nCOUNTERFACTUALS ROBUST: {len(cfr)}")
    for j, cf in enumerate(cfr, start=1):
        print("\n" + "-" * 60)
        print(f"[ROBUST #{j}] type={cf.get('type')} status={cf.get('status')} strict={cf.get('strict_valid')}")
        print("term:", cf.get("term"))
        print("replacement:", cf.get("replacement"))
        print("ant_subtype:", cf.get("ant_subtype"))
        print("ungrammatical_proxy:", cf.get("ungrammatical_proxy"))
        print("tfidf_weight_raw:", cf.get("tfidf_weight_raw"), "tfidf_weight_norm:", cf.get("tfidf_weight_norm"))
        print("modified_input:\n", cf.get("modified_input"))
        print("final_advice:\n", cf.get("final_advice"))
        print("nli_proxy:", cf.get("nli_proxy"))
        print("impact_1_minus_bert:", cf.get("impact_1_minus_bert"))
        if PRINT_CF_TEXT:
            txt = cf.get("text", "") or ""
            print("\ncf_text:\n", _shorten(txt, PRINT_MAX_CF_CHARS))

    print(f"\nCOUNTERFACTUALS STRICT: {len(cfs)}")
    for j, cf in enumerate(cfs, start=1):
        print("\n" + "-" * 60)
        print(f"[STRICT #{j}] type={cf.get('type')}")
        print("term:", cf.get("term"))
        print("replacement:", cf.get("replacement"))
        print("ant_subtype:", cf.get("ant_subtype"))
        print("tfidf_weight_raw:", cf.get("tfidf_weight_raw"), "tfidf_weight_norm:", cf.get("tfidf_weight_norm"))
        print("modified_input:\n", cf.get("modified_input"))
        print("final_advice:\n", cf.get("final_advice"))
        print("nli_proxy:", cf.get("nli_proxy"))
        print("impact_1_minus_bert:", cf.get("impact_1_minus_bert"))


#safe timeout for blocking calls (Ollama)
class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError()

def call_with_timeout(fn, seconds: int, *args, **kwargs):
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


#bounded LRU cache
class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = int(maxsize)
        self._d: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, k: str):
        if k not in self._d:
            return None
        self._d.move_to_end(k)
        return self._d[k]

    def set(self, k: str, v: Any):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)



#LLM (Ollama)
def _build_llm() -> OllamaLLM:
    try:
        if OLLAMA_KEEP_ALIVE:
            return OllamaLLM(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, keep_alive=OLLAMA_KEEP_ALIVE)
    except Exception:
        pass
    return OllamaLLM(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

llm = _build_llm()

def timed_llm_invoke(name: str, chain: LLMChain, payload: Dict[str, Any]) -> Any:
    t0 = time.time()
    out = call_with_timeout(chain.invoke, LLM_TIMEOUT_SEC, payload)
    if PRINT_TIMERS:
        print(f"[TIMER][LLM] {name} took {time.time() - t0:.2f}s")
    return out


#helpers functions
def _get_invoke_text(resp: Any, preferred_key: Optional[str] = None) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        if preferred_key and preferred_key in resp and resp[preferred_key] is not None:
            return str(resp[preferred_key])
        for k in ["text", "output", "result", "content"]:
            if k in resp and resp[k] is not None:
                return str(resp[k])
        return str(resp)
    return str(resp)

def safe_parse_json(raw: Any, key: str, default: str) -> str:
    try:
        if isinstance(raw, dict) and key in raw:
            v = str(raw.get(key, "")).strip()
            return v if v else default
        s = str(raw or "")
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            data = json.loads(m.group())
            v = data.get(key, "")
            if v and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    return default

def _norm_phrase(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

#soft match normalization (punctuation tolerant)
def _normalize_for_soft_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

#tighter boundaries (avoid \w / underscore quirks)
def _phrase_regex(phrase: str) -> str:
    toks = [re.escape(t) for t in (phrase or "").strip().split()]
    if not toks:
        return ""
    joined = r"\s+".join(toks)
    return rf"(?i)(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])"

#exact-regex OR soft-match fallback
def term_occurs_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    pat = _phrase_regex(term)
    if pat and re.search(pat, text):
        return True
    return _normalize_for_soft_match(term) in _normalize_for_soft_match(text)

def replace_phrase_in_text(text: str, phrase: str, replacement: str) -> Tuple[str, bool]:
    if not text or not phrase:
        return text, False
    pat = _phrase_regex(phrase)
    if pat and re.search(pat, text) is not None:
        new_text = re.sub(pat, replacement, text, count=1)
        return new_text, new_text != text

    #fallback:soft match replacement
    src = _normalize_for_soft_match(text)
    ph  = _normalize_for_soft_match(phrase)
    if not ph or ph not in src:
        return text, False

    toks = [re.escape(t) for t in phrase.strip().split()]
    if not toks:
        return text, False
    loose = r"(?i)(?<![A-Za-z0-9])" + r"[^\w]*\s*".join(toks) + r"(?![A-Za-z0-9])"
    if re.search(loose, text) is None:
        return text, False
    new_text = re.sub(loose, replacement, text, count=1)
    return new_text, new_text != text

def ungrammatical_proxy(modified_input: str) -> bool:
    if not modified_input:
        return False
    t = modified_input.lower()
    if "  " in modified_input:
        return True
    if re.search(r"\b(and|or)\s*[,.]", t):
        return True
    if re.search(r"\bis\s+no\s+\w+", t):
        return True
    return False

#sanitizer:remove trailing explanations only if pattern looks like "TERM definition"/keep slashes as-is (NOT forcing removal)/allow up to 6 words (less truncation)
def sanitize_influential_term(term: str) -> str:
  
    t = (term or "").strip()

    #strip definition-like tail
    for sep in [" - ", " – ", " — ", ": "]:
        if sep in t and len(t.split(sep, 1)[0].split()) <= 6:
            t = t.split(sep, 1)[0].strip()
            break

    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > 6:
        t = " ".join(words[:6]).strip()
    return t




#TF-IDF global&cosine&weights
_TFIDF_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF_VEC_CACHE = LRUCache(maxsize=TFIDF_VEC_CACHE_MAX)

def build_tfidf_vectorizer(corpus_texts: List[str]) -> TfidfVectorizer:
    vect = TfidfVectorizer(
        lowercase=True,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        token_pattern=r"(?u)\b\w+\b",
    )
    vect.fit(corpus_texts if corpus_texts else [""])
    return vect

def _tfidf_vec(text: str):
    global _TFIDF_VECTORIZER
    key = str(text or "")
    v = _TFIDF_VEC_CACHE.get(key)
    if v is None:
        v = _TFIDF_VECTORIZER.transform([key])
        _TFIDF_VEC_CACHE.set(key, v)
    return v

def tfidf_cosine(a: str, b: str) -> float:
    global _TFIDF_VECTORIZER
    if _TFIDF_VECTORIZER is None:
        return 0.0
    va = _tfidf_vec(a or "")
    vb = _tfidf_vec(b or "")
    if va.nnz == 0 and vb.nnz == 0:
        return 0.0
    return float(sk_cosine_similarity(va, vb)[0, 0])

def tfidf_weights_for_terms(question_text: str, terms: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    global _TFIDF_VECTORIZER
    if _TFIDF_VECTORIZER is None or not terms:
        raw0 = {t: 0.0 for t in (terms or [])}
        return raw0, raw0

    vec = _TFIDF_VECTORIZER.transform([question_text or ""])
    if vec.nnz == 0:
        raw0 = {t: 0.0 for t in terms}
        return raw0, raw0

    feature_names = _TFIDF_VECTORIZER.get_feature_names_out()
    feat2idx = {feature_names[i]: i for i in range(len(feature_names))}
    row = vec.tocsr()[0]
    idx2val = {int(i): float(v) for i, v in zip(row.indices, row.data)}

    raw: Dict[str, float] = {}
    for t in terms:
        tn = _norm_phrase(t)
        if not tn:
            raw[t] = 0.0
            continue
        if tn in feat2idx:
            raw[t] = float(idx2val.get(int(feat2idx[tn]), 0.0))
        else:
            best = 0.0
            for tok in tn.split():
                if tok in feat2idx:
                    best = max(best, float(idx2val.get(int(feat2idx[tok]), 0.0)))
            raw[t] = float(best)

    s = float(sum(raw.values())) or 1.0
    norm = {k: float(v) / s for k, v in raw.items()}
    return raw, norm




#PROMPTS
#influentials must be EXACT substrings from QUESTION
influential_prompt = PromptTemplate.from_template(r"""
Return ONLY a valid Python list of exactly 2 strings. No extra text.

Rules (must follow):
- Each string MUST be an EXACT substring copied from the QUESTION (verbatim).
- 1 to 4 words per string.
- Do NOT add explanations/definitions.
- Do NOT add slashes like "A/B".
- Prefer symptoms/conditions/medications that strongly affect advice.

QUESTION:
{input}
""")

synonym_prompt = PromptTemplate.from_template(r"""
You are given ONE influential medical term: "{word}".

Task:
- Generate ONE medically valid synonym (common alternative term).
- If no valid synonym exists, return "no change".

Return ONLY valid JSON:
{{
  "synonym": "..."
}}
""")

antonym_prompt = PromptTemplate.from_template(r"""
You are given ONE influential medical term: "{word}".

Task:
- Return ONE medically plausible OPPOSITE / ANTONYM if it exists as a short phrase.
- If no true antonym exists, return "NO_ANTONYM".

Return ONLY valid JSON:
{{
  "antonym": "..."
}}
""")

negation_prompt = PromptTemplate.from_template(r"""
You are given ONE influential medical term: "{word}".

Task:
- Produce ONE medically plausible NEGATION of the term (absence/denial).
- Output should be SHORT (1–5 words) suitable to replace the original inside a sentence.

Examples:
- "fever" -> "no fever"
- "chest pain" -> "no chest pain"
- "metformin" -> "not taking metformin"

Return ONLY valid JSON:
{{
  "negation": "..."
}}
""")

counterfactual_body_prompt = PromptTemplate.from_template(r"""
You are a careful medical assistant.

Output MUST contain ONLY these 2 sections in order (no extra text):

Modified Input:
"..."

Final Advice:
"..."

Constraints:
- Copy the Modified Input EXACTLY as provided (verbatim inside quotes).
- Final Advice MUST be concrete and in quotes (1–3 sentences).
- Do NOT include reasoning.

Modified Input:
{modified_input}
""")

influential_chain = LLMChain(llm=llm, prompt=influential_prompt, output_key="influential_words")
synonym_chain = LLMChain(llm=llm, prompt=synonym_prompt, output_key="synonym")
antonym_chain = LLMChain(llm=llm, prompt=antonym_prompt, output_key="antonym")
negation_chain = LLMChain(llm=llm, prompt=negation_prompt, output_key="negation")
counterfactual_chain = LLMChain(llm=llm, prompt=counterfactual_body_prompt, output_key="counterfactual")



#signature/hash
def make_prompt_signature() -> str:
    parts = {
        "MODE_TO_RUN": MODE_TO_RUN,
        "LLM_MODEL_NAME": LLM_MODEL_NAME,
        "LLM_TEMPERATURE": str(LLM_TEMPERATURE),
        "MAX_RETRIES": str(MAX_RETRIES),
        "INFLUENTIAL": influential_prompt.template,
        "SYN": synonym_prompt.template,
        "ANT": antonym_prompt.template,
        "NEG": negation_prompt.template,
        "CF": counterfactual_body_prompt.template,
        "TFIDF": f"{TFIDF_NGRAM_RANGE}|{TFIDF_MIN_DF}|{TFIDF_MAX_FEATURES}",
        "EVAL_BERT": str(EVAL_BERT),
        "EVAL_NLI": str(EVAL_NLI),
        "EVAL_ROUGE": str(EVAL_ROUGE),
        "EVAL_METEOR": str(EVAL_METEOR),
        "SEED": str(SEED),
        "DEVICE": str(DEVICE),
    }
    blob = "\n".join([f"{k}={parts[k]}" for k in sorted(parts.keys())])
    return hashlib.md5(blob.encode("utf-8")).hexdigest()

PROMPT_SIG = make_prompt_signature()

def compute_sample_hash(sample: Dict[str, str]) -> str:
    combined = (
        f"{MODE_TO_RUN}||{LLM_MODEL_NAME}||{LLM_TEMPERATURE}||{PROMPT_SIG}||"
        f"{sample.get('instruction','')}||{sample.get('input','')}||{sample.get('output','')}"
    )
    return hashlib.md5(combined.encode("utf-8")).hexdigest()



#influential parsing
def _parse_python_list_of_strings(raw: str) -> List[str]:
    s = (raw or "").strip()
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return []
    items = re.findall(r"""['"]([^'"]+)['"]""", m.group(0))
    out, seen = [], set()
    for it in items:
        itn = _norm_phrase(it)
        if not itn or itn in seen:
            continue
        seen.add(itn)
        out.append(it.strip())
    return out

def _parse_numbered_list(raw: str) -> List[str]:
    s = str(raw or "")
    out, seen = [], set()
    for line in s.splitlines():
        m = re.match(r"^\s*\d+\s*[\.\)]\s*(.+?)\s*$", line)
        if m:
            t = m.group(1).strip()
            tn = _norm_phrase(t)
            if t and tn not in seen:
                seen.add(tn)
                out.append(t)
    return out

def extract_influentials_robust(raw: str) -> List[str]:
    terms = _parse_python_list_of_strings(raw)
    if terms:
        return terms[:INFLUENTIAL_TOPK_MAX]
    terms2 = _parse_numbered_list(raw)
    if terms2:
        return terms2[:INFLUENTIAL_TOPK_MAX]
    return []


#strict validation for CF output
_PLACEHOLDER_MARKERS = {"...", "…", "tbd", "todo", "your final advice here", "final advice here"}

def _is_bad_final_advice(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    tn = _norm_phrase(t)
    if tn in _PLACEHOLDER_MARKERS:
        return True
    if re.fullmatch(r"[.\u2026\-\s]+", t):
        return True
    if len(t) < 12:
        return True
    if re.search(r"[A-Za-z]", t) is None:
        return True
    return False

def extract_final_advice(cf_text: str) -> str:
    if not cf_text:
        return ""
    m = re.search(r'(?is)\bfinal\s+advice\s*:\s*"(.*?)"\s*(?:\n|$)', cf_text)
    if m:
        fa = (m.group(1) or "").strip()
        return "" if _is_bad_final_advice(fa) else fa
    return ""

def is_strict_counterfactual(cf_text: str) -> bool:
    if not cf_text:
        return False
    has_mod = re.search(r'(?is)\bmodified\s+input\s*:\s*"', cf_text) is not None
    has_adv = re.search(r'(?is)\bfinal\s+advice\s*:\s*"', cf_text) is not None
    if not (has_mod and has_adv):
        return False
    return bool(extract_final_advice(cf_text))

def _inject_header(term: str, replacement: str, cf_body: str) -> str:
    return f'--- Counterfactual for "{term}" (Replacement: "{replacement}") ---\n' + (cf_body or "").lstrip()



# Metrics
rouge_scorer_fn = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_BERT_MODEL = None
_EMB_CACHE = LRUCache(maxsize=EMB_CACHE_MAX)

def _get_bert_model():
    global _BERT_MODEL
    if _BERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        if PRINT_LOAD_LINES:
            print(f"[LOAD] SentenceTransformer: {BERT_MODEL_PATH} on {DEVICE}")
        _BERT_MODEL = SentenceTransformer(BERT_MODEL_PATH, device=DEVICE)
    return _BERT_MODEL

def _bert_embed(text: str) -> torch.Tensor:
    key = str(text or "")
    v = _EMB_CACHE.get(key)
    if v is not None:
        return v
    model = _get_bert_model()
    with torch.inference_mode():
        emb = model.encode([key], convert_to_tensor=True, show_progress_bar=False)[0]
    _EMB_CACHE.set(key, emb)
    return emb

def cosine_bert_from_emb(ea: torch.Tensor, eb: torch.Tensor) -> float:
    try:
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))
    except Exception:
        return 0.0

def effect_size_from_emb(ea: torch.Tensor, eb: torch.Tensor) -> float:
    c = cosine_bert_from_emb(ea, eb)
    return float(min(1.0, max(0.0, 1.0 - c)))

def rouge_l(a: str, b: str) -> float:
    if not EVAL_ROUGE:
        return 0.0
    try:
        return rouge_scorer_fn.score(str(a or ""), str(b or ""))["rougeL"].fmeasure
    except Exception:
        return 0.0

_METEOR_FAILURES = 0
def meteor(a: str, b: str) -> float:
    global _METEOR_FAILURES
    if not EVAL_METEOR:
        return 0.0
    aa = str(a or "").strip()
    bb = str(b or "").strip()
    if not aa or not bb:
        return 0.0
    try:
        return meteor_score([bb.split()], aa.split())
    except Exception:
        _METEOR_FAILURES += 1
        return 0.0



#NLI proxy&cache
_NLI_TOKENIZER = None
_NLI_MODEL = None
_ID2LABEL: Optional[Dict[int, str]] = None

def _get_nli():
    global _NLI_TOKENIZER, _NLI_MODEL, _ID2LABEL
    if not EVAL_NLI:
        return None, None, None
    if _NLI_TOKENIZER is None or _NLI_MODEL is None or _ID2LABEL is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if PRINT_LOAD_LINES:
            print(f"[LOAD] NLI model: {NLI_MODEL_NAME} on {DEVICE}")
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE)
        _NLI_MODEL.eval()
        _ID2LABEL = {int(k): str(v).upper() for k, v in _NLI_MODEL.config.id2label.items()}
    return _NLI_TOKENIZER, _NLI_MODEL, _ID2LABEL

def nli_proxy_label(premise: str, hypothesis: str) -> str:
    if not EVAL_NLI:
        return "DISABLED"
    try:
        tok, model, id2label = _get_nli()
        if tok is None or model is None or id2label is None:
            return "DISABLED"
        enc = tok(str(premise or ""), str(hypothesis or ""), return_tensors="pt",
                  truncation=True, max_length=NLI_MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.inference_mode():
            pred_id = int(torch.argmax(model(**enc).logits[0]).item())
        lbl = id2label.get(pred_id, "UNKNOWN")
        if "ENTAIL" in lbl:
            return "ENTAILMENT"
        if "CONTRAD" in lbl:
            return "CONTRADICTION"
        if "NEUT" in lbl:
            return "NEUTRAL"
        return lbl
    except Exception:
        return "UNKNOWN"

@lru_cache(maxsize=NLI_LRU_MAX)
def nli_proxy_label_cached(premise: str, hypothesis: str) -> str:
    return nli_proxy_label(premise, hypothesis)




#cached LLM calls
@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_influentials(question: str) -> Tuple[str, ...]:
    try:
        resp = timed_llm_invoke("influential", influential_chain, {"input": str(question or "")})
    except _TimeoutError:
        return tuple()
    raw = _get_invoke_text(resp, preferred_key="influential_words")
    terms = extract_influentials_robust(raw)
    return tuple(terms[:INFLUENTIAL_TOPK_MAX])

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_synonym(term: str) -> str:
    try:
        resp = timed_llm_invoke("synonym", synonym_chain, {"word": str(term or "")})
    except _TimeoutError:
        return "no change"
    raw = _get_invoke_text(resp, preferred_key="synonym")
    return safe_parse_json(raw, "synonym", "no change")

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_antonym(term: str) -> str:
    try:
        resp = timed_llm_invoke("antonym", antonym_chain, {"word": str(term or "")})
    except _TimeoutError:
        return "NO_ANTONYM"
    raw = _get_invoke_text(resp, preferred_key="antonym")
    return safe_parse_json(raw, "antonym", "NO_ANTONYM")

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_negation(term: str) -> str:
    try:
        resp = timed_llm_invoke("negation", negation_chain, {"word": str(term or "")})
    except _TimeoutError:
        return f"no {term}"
    raw = _get_invoke_text(resp, preferred_key="negation")
    return safe_parse_json(raw, "negation", f"no {term}")

def _synonym_is_usable(term: str, syn: str) -> bool:
    sn = _norm_phrase(syn)
    if not sn or sn == "no change":
        return False
    if sn == _norm_phrase(term):
        return False
    if re.search(r"[.!?]", syn or ""):
        return False
    return True

def _antonym_is_usable(term: str, ant: str) -> bool:
    an = _norm_phrase(ant)
    if not an:
        return False
    if an in {"no_antonym", "no antonym", "no antonym exists", "no_antonm"}:
        return False
    if an == _norm_phrase(term):
        return False
    return True


def choose_ant_or_neg(term: str) -> Tuple[str, str]:
    ant = _cached_antonym(term)
    if _antonym_is_usable(term, ant):
        return ant, "antonym"
    neg = _cached_negation(term)
    neg = neg.strip() if isinstance(neg, str) else ""
    return (neg if neg else f"no {term}"), "negation"

def types_for_mode(mode: str) -> List[str]:
    m = (mode or "").strip().upper()
    if m == "SYN_ONLY":
        return ["syn"]
    if m == "ANT_ONLY":
        return ["ant"]
    return ["syn", "ant"]




#METEOR
def nltk_preflight():
    if not EVAL_METEOR:
        return
    import nltk
    from nltk.data import find

    if "/Users/agapikyrimi/nltk_data" not in nltk.data.path:
        nltk.data.path.insert(0, "/Users/agapikyrimi/nltk_data")

    def _exists_any(candidates: List[str]) -> bool:
        for c in candidates:
            try:
                find(c)
                return True
            except LookupError:
                continue
        return False

    ok_wordnet = _exists_any(["corpora/wordnet", "corpora/wordnet.zip"])
    ok_omw = _exists_any(["corpora/omw-1.4", "corpora/omw-1.4.zip"])

    if not (ok_wordnet and ok_omw):
        print(
            "[NLTK] wordnet missing. METEOR may be slow/fail.\n"
            "Run once:\n"
            "  python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        )




#core: process one sample
def process_sample(sample: Dict[str, str], mode: str = MODE_TO_RUN) -> Dict[str, Any]:
    orig_in = str(sample.get("input", "") or "")
    orig_out = str(sample.get("output", "") or "")

    #influentials
    terms_raw = list(_cached_influentials(orig_in))

    #sanitize&keep only those that really occur in text
    terms: List[str] = []
    for t in terms_raw:
        ts = sanitize_influential_term(t)
        if ts and term_occurs_in_text(ts, orig_in):
            terms.append(ts)
        elif t and term_occurs_in_text(t, orig_in):
            terms.append(t)

    
    terms = terms[:INFLUENTIAL_TOPK_MAX]

    w_raw, w_norm = tfidf_weights_for_terms(orig_in, terms)

    results: Dict[str, Any] = {
        "original_input": orig_in,
        "original_output": orig_out,
        "term_candidates_influential": terms,
        "tfidf_weights_raw": w_raw,
        "tfidf_weights_norm": w_norm,
        "counterfactuals_robust": [],
        "counterfactuals_strict": [],
        "per_type_status": {},
    }

    for cf_type in types_for_mode(mode):
        robust = {
            "type": cf_type,
            "term": None,
            "replacement": None,
            "modified_input": None,
            "text": "",
            "final_advice": "",
            "strict_valid": False,
            "impact_1_minus_bert": None,
            "nli_proxy": "UNKNOWN",
            "status": "NOT_ATTEMPTED",
            "ungrammatical_proxy": False,
            "tfidf_weight_raw": 0.0,
            "tfidf_weight_norm": 0.0,
            "ant_subtype": None,
        }

        chosen_term = None
        replacement = None
        modified_input = None
        ant_subtype = None

        for term in terms:
            #never call synonym/antonym if term not in input
            if not term_occurs_in_text(term, orig_in):
                continue

            if cf_type == "syn":
                syn = _cached_synonym(term)
                if not _synonym_is_usable(term, syn):
                    continue
                new_in, did = replace_phrase_in_text(orig_in, term, syn)
                if not did:
                    continue
                chosen_term, replacement, modified_input = term, syn, new_in
                break

            if cf_type == "ant":
                rep, sub = choose_ant_or_neg(term)
                new_in, did = replace_phrase_in_text(orig_in, term, rep)
                if not did:
                    continue
                chosen_term, replacement, modified_input = term, rep, new_in
                ant_subtype = sub
                break

        if chosen_term is None:
            robust["status"] = "NO_MATCH" if terms else "NO_INFLUENTIAL_TERMS"
            results["counterfactuals_robust"].append(robust)
            results["per_type_status"][cf_type] = robust["status"]
            continue

        robust["ungrammatical_proxy"] = bool(ungrammatical_proxy(modified_input))

        cf_body = ""
        for _ in range(max(1, int(MAX_RETRIES))):
            try:
                resp = timed_llm_invoke(
                    f"counterfactual[{cf_type}]",
                    counterfactual_chain,
                    {"modified_input": str(modified_input or "")},
                )
            except _TimeoutError:
                resp = ""
            cf_body = _get_invoke_text(resp, preferred_key="counterfactual") or ""
            if cf_body.strip():
                break

        cf_text = _inject_header(chosen_term, replacement, cf_body)

        robust.update({
            "term": chosen_term,
            "replacement": replacement,
            "modified_input": modified_input,
            "text": cf_text,
            "status": "CF_GENERATED",
            "tfidf_weight_raw": float(w_raw.get(chosen_term, 0.0)),
            "tfidf_weight_norm": float(w_norm.get(chosen_term, 0.0)),
        })
        if cf_type == "ant":
            robust["ant_subtype"] = ant_subtype

        final_advice = str(extract_final_advice(cf_text) or "")
        strict_ok = is_strict_counterfactual(cf_text)

        robust["strict_valid"] = bool(strict_ok)
        robust["final_advice"] = final_advice

        if not final_advice:
            robust["status"] = "NO_FINAL_ADVICE"
            results["counterfactuals_robust"].append(robust)
            results["per_type_status"][cf_type] = robust["status"]
            continue

        robust["nli_proxy"] = str(nli_proxy_label_cached(orig_out, final_advice))

        if strict_ok:
            results["counterfactuals_strict"].append({
                "type": cf_type,
                "term": chosen_term,
                "replacement": replacement,
                "modified_input": modified_input,
                "text": cf_text,
                "final_advice": final_advice,
                "impact_1_minus_bert": None,
                "nli_proxy": robust["nli_proxy"],
                "tfidf_weight_raw": float(w_raw.get(chosen_term, 0.0)),
                "tfidf_weight_norm": float(w_norm.get(chosen_term, 0.0)),
                "ant_subtype": ant_subtype if cf_type == "ant" else None,
            })

        results["counterfactuals_robust"].append(robust)
        results["per_type_status"][cf_type] = robust["status"]

    return results




#error analysis aggregators
def _init_error_counters(types: List[str]):
    return {
        "status_counts": {t: Counter() for t in types},
        "strict_valid": {t: Counter() for t in types},
        "ungrammatical": {t: Counter() for t in types},
        "ant_subtype": Counter(),
        "attempts": Counter(),
    }

def _safe_rate(num: int, den: int) -> Optional[float]:
    return float(num) / float(den) if den else None

def _finalize_error_analysis(counters, types: List[str]) -> Dict[str, Any]:
    counts_per_status_per_type = {t: dict(counters["status_counts"][t]) for t in types}
    strict_valid_rate_per_type = {
        t: _safe_rate(counters["strict_valid"][t]["true"], counters["attempts"][t]) for t in types
    }
    ungrammatical_rate_per_type = {
        t: _safe_rate(counters["ungrammatical"][t]["true"], counters["attempts"][t]) for t in types
    }
    no_match_rate_per_type = {
        t: _safe_rate(
            counters["status_counts"][t]["NO_MATCH"] + counters["status_counts"][t]["NO_INFLUENTIAL_TERMS"],
            counters["attempts"][t]
        )
        for t in types
    }
    no_final_advice_rate_per_type = {
        t: _safe_rate(counters["status_counts"][t]["NO_FINAL_ADVICE"], counters["attempts"][t]) for t in types
    }

    ant_total = sum(counters["ant_subtype"].values())
    ant_subtype_share = {
        "antonym": _safe_rate(counters["ant_subtype"]["antonym"], ant_total),
        "negation": _safe_rate(counters["ant_subtype"]["negation"], ant_total),
        "counts": dict(counters["ant_subtype"]),
    }

    return {
        "counts_per_status_per_type": counts_per_status_per_type,
        "strict_valid_rate_per_type": strict_valid_rate_per_type,
        "ant_subtype_share": ant_subtype_share,
        "ungrammatical_rate_per_type": ungrammatical_rate_per_type,
        "no_match_rate_per_type": no_match_rate_per_type,
        "no_final_advice_rate_per_type": no_final_advice_rate_per_type,
        "attempts_per_type": dict(counters["attempts"]),
    }

def error_analysis_from_cache(stored: Dict[str, Any], mode: str) -> Dict[str, Any]:
    types = types_for_mode(mode)
    counters = _init_error_counters(types)

    for _h, obj in stored.items():
        res = (obj or {}).get("result") or {}
        for cf in res.get("counterfactuals_robust", []) or []:
            typ = (cf.get("type") or "").strip()
            if typ not in types:
                continue
            counters["attempts"][typ] += 1
            st = str(cf.get("status") or "UNKNOWN")
            counters["status_counts"][typ][st] += 1
            counters["strict_valid"][typ]["true" if bool(cf.get("strict_valid")) else "false"] += 1
            counters["ungrammatical"][typ]["true" if bool(cf.get("ungrammatical_proxy")) else "false"] += 1
            if typ == "ant":
                sub = str(cf.get("ant_subtype") or "").strip().lower()
                if sub in {"antonym", "negation"}:
                    counters["ant_subtype"][sub] += 1

    return _finalize_error_analysis(counters, types)




#evaluation
def evaluate_dataset(samples: List[Dict[str, str]], max_items: int = MAX_ITEMS, mode: str = MODE_TO_RUN) -> Dict[str, Any]:
    def new_out_bucket():
        return {"cosine": [], "seq": [], "bert": [], "rougeL": [], "meteor": [], "nli": [], "impact": []}

    def new_in_bucket():
        return {"cosine": [], "seq": [], "bert": [], "rougeL": [], "meteor": []}

    out_keys = ["syn_out_vs_orig_out", "ant_out_vs_orig_out"]
    in_keys  = ["syn_in_vs_orig_in",  "ant_in_vs_orig_in"]

    outR = {k: new_out_bucket() for k in out_keys}
    outS = {k: new_out_bucket() for k in out_keys}
    inR  = {k: new_in_bucket()  for k in in_keys}
    inS  = {k: new_in_bucket()  for k in in_keys}

    types = types_for_mode(mode)
    counters = _init_error_counters(types)

    total = min(len(samples), max_items)

    if PRINT_EVAL_START:
        print("[EVAL] start evaluate_dataset")

    for i, sample in enumerate(samples[:max_items]):
        idx1 = i + 1
        h = compute_sample_hash(sample)

        res = None
        if h in stored_results:
            res = stored_results[h].get("result")

        if res is None:
            try:
                res = process_sample(sample, mode=mode)
            except Exception as e:
             
                err_msg = str(e)

                err_cfs: List[Dict[str, Any]] = []
                for typ in types_for_mode(mode):
                    err_cfs.append({
                        "type": typ,
                        "status": "PROCESS_SAMPLE_EXCEPTION",
                        "error": err_msg,
                        "strict_valid": False,
                        "ungrammatical_proxy": False,
                        "ant_subtype": None,
                        "term": None,
                        "replacement": None,
                        "modified_input": None,
                        "text": "",
                        "final_advice": "",
                        "impact_1_minus_bert": None,
                        "nli_proxy": "UNKNOWN",
                        "tfidf_weight_raw": 0.0,
                        "tfidf_weight_norm": 0.0,
                    })

                stored_results[h] = {
                    "result": {
                        "original_input": str(sample.get("input", "") or ""),
                        "original_output": str(sample.get("output", "") or ""),
                        "term_candidates_influential": [],
                        "tfidf_weights_raw": {},
                        "tfidf_weights_norm": {},
                        "counterfactuals_robust": err_cfs,
                        "counterfactuals_strict": [],
                        "per_type_status": {cf["type"]: "PROCESS_SAMPLE_EXCEPTION" for cf in err_cfs},
                    },
                    "meta": {"prompt_sig": PROMPT_SIG},
                }

                if (idx1) % CACHE_FLUSH_EVERY == 0:
                    try:
                        with open(CACHE_FILE, "wb") as f:
                            pickle.dump(stored_results, f)
                    except Exception:
                        pass
                continue

            stored_results[h] = {"result": res, "meta": {"prompt_sig": PROMPT_SIG}}

       
        if PRINT_EVERY and (idx1 % PRINT_EVERY == 0):
            _print_sample(res, idx1, total, h)

        orig_out = str(res.get("original_output", "") or "")
        orig_in  = str(res.get("original_input", "") or "")

        orig_out_emb = _bert_embed(orig_out) if EVAL_BERT else None
        orig_in_emb  = _bert_embed(orig_in)  if EVAL_BERT else None

        strict_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for scf in res.get("counterfactuals_strict", []) or []:
            k = (str(scf.get("type")), str(scf.get("term")), str(scf.get("replacement")))
            strict_index[k] = scf

        cache_dirty = False

        #counters
        for cf in res.get("counterfactuals_robust", []) or []:
            typ = (cf.get("type") or "").strip()
            if typ not in types:
                continue
            counters["attempts"][typ] += 1
            st = str(cf.get("status") or "UNKNOWN")
            counters["status_counts"][typ][st] += 1
            counters["strict_valid"][typ]["true" if bool(cf.get("strict_valid")) else "false"] += 1
            counters["ungrammatical"][typ]["true" if bool(cf.get("ungrammatical_proxy")) else "false"] += 1
            if typ == "ant":
                sub = str(cf.get("ant_subtype") or "").strip().lower()
                if sub in {"antonym", "negation"}:
                    counters["ant_subtype"][sub] += 1

        #metrics
        for cf in res.get("counterfactuals_robust", []) or []:
            typ = (cf.get("type") or "").strip()
            key_out = f"{typ}_out_vs_orig_out"
            key_in  = f"{typ}_in_vs_orig_in"

            cf_out = str((cf.get("final_advice") or "").strip())
            is_strict = bool(cf.get("strict_valid"))

            if cf_out and key_out in outR:
                if EVAL_BERT and orig_out_emb is not None:
                    cf_out_emb = _bert_embed(cf_out)
                    b = cosine_bert_from_emb(cf_out_emb, orig_out_emb)
                    imp = effect_size_from_emb(cf_out_emb, orig_out_emb)
                else:
                    b, imp = 0.0, None

                if imp is not None and cf.get("impact_1_minus_bert") != float(imp):
                    cf["impact_1_minus_bert"] = float(imp)
                    cache_dirty = True

                outR[key_out]["cosine"].append(tfidf_cosine(cf_out, orig_out))
                outR[key_out]["seq"].append(SequenceMatcher(None, cf_out, orig_out).ratio())
                outR[key_out]["bert"].append(b)
                outR[key_out]["rougeL"].append(rouge_l(cf_out, orig_out))
                outR[key_out]["meteor"].append(meteor(cf_out, orig_out))
                outR[key_out]["nli"].append(cf.get("nli_proxy", "UNKNOWN"))
                if imp is not None:
                    outR[key_out]["impact"].append(float(imp))

                if is_strict and key_out in outS:
                    outS[key_out]["cosine"].append(outR[key_out]["cosine"][-1])
                    outS[key_out]["seq"].append(outR[key_out]["seq"][-1])
                    outS[key_out]["bert"].append(outR[key_out]["bert"][-1])
                    outS[key_out]["rougeL"].append(outR[key_out]["rougeL"][-1])
                    outS[key_out]["meteor"].append(outR[key_out]["meteor"][-1])
                    outS[key_out]["nli"].append(outR[key_out]["nli"][-1])
                    if imp is not None:
                        outS[key_out]["impact"].append(float(imp))

                    k = (str(cf.get("type")), str(cf.get("term")), str(cf.get("replacement")))
                    scf = strict_index.get(k)
                    if scf is not None and imp is not None and scf.get("impact_1_minus_bert") != float(imp):
                        scf["impact_1_minus_bert"] = float(imp)
                        cache_dirty = True

            mod_in = str((cf.get("modified_input") or "").strip())
            if mod_in and key_in in inR:
                if EVAL_BERT and orig_in_emb is not None:
                    mod_in_emb = _bert_embed(mod_in)
                    b_in = cosine_bert_from_emb(mod_in_emb, orig_in_emb)
                else:
                    b_in = 0.0

                inR[key_in]["cosine"].append(tfidf_cosine(mod_in, orig_in))
                inR[key_in]["seq"].append(SequenceMatcher(None, mod_in, orig_in).ratio())
                inR[key_in]["bert"].append(b_in)
                inR[key_in]["rougeL"].append(rouge_l(mod_in, orig_in))
                inR[key_in]["meteor"].append(meteor(mod_in, orig_in))

                if is_strict and key_in in inS:
                    inS[key_in]["cosine"].append(inR[key_in]["cosine"][-1])
                    inS[key_in]["seq"].append(inR[key_in]["seq"][-1])
                    inS[key_in]["bert"].append(inR[key_in]["bert"][-1])
                    inS[key_in]["rougeL"].append(inR[key_in]["rougeL"][-1])
                    inS[key_in]["meteor"].append(inR[key_in]["meteor"][-1])

        if cache_dirty:
            stored_results[h] = {"result": res, "meta": {"prompt_sig": PROMPT_SIG}}

        if (idx1) % CACHE_FLUSH_EVERY == 0:
            try:
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(stored_results, f)
            except Exception:
                pass

    #final
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(stored_results, f)
    except Exception:
        pass

    def mean(x): return float(np.mean(x)) if x else None
    def majority(x): return max(set(x), key=x.count) if x else None

    def pack_out(b):
        return {
            "cosine": mean(b["cosine"]),
            "seq": mean(b["seq"]),
            "bert": mean(b["bert"]),
            "rougeL": mean(b["rougeL"]),
            "meteor": mean(b["meteor"]),
            "nli_proxy_majority": majority(b["nli"]),
            "impact_mean": mean(b["impact"]),
        }

    def pack_in(b):
        return {
            "cosine": mean(b["cosine"]),
            "seq": mean(b["seq"]),
            "bert": mean(b["bert"]),
            "rougeL": mean(b["rougeL"]),
            "meteor": mean(b["meteor"]),
        }

    error_analysis = _finalize_error_analysis(counters, types)

    return {
        "robust_out": {k: pack_out(v) for k, v in outR.items()},
        "strict_out": {k: pack_out(v) for k, v in outS.items()},
        "robust_in":  {k: pack_in(v)  for k, v in inR.items()},
        "strict_in":  {k: pack_in(v)  for k, v in inS.items()},
        "meta": {
            "mode": mode,
            "device": str(DEVICE),
            "seed": int(SEED),
            "cache_file": CACHE_FILE,
            "prompt_sig": PROMPT_SIG,
            "max_items": int(max_items),
            "eval_bert": bool(EVAL_BERT),
            "eval_nli": bool(EVAL_NLI),
            "eval_rouge": bool(EVAL_ROUGE),
            "eval_meteor": bool(EVAL_METEOR),
            "versions": get_versions(),
            "error_analysis": error_analysis,
        },
    }




#main
if __name__ == "__main__":
    nltk_preflight()

    with open("cleaned_healthcaremagic.pkl", "rb") as f:
        cleaned_healthcaremagic = pickle.load(f)

    print(f"[DATA] Loaded dataset with {len(cleaned_healthcaremagic)} samples")
    print(f"[META] seed={SEED} device={DEVICE} versions={get_versions()}")
    print(f"[CACHE] {CACHE_FILE}")
    rb = rss_bytes()
    if rb is not None:
        print(f"[BOOT] RSS={rb/1024/1024:.1f}MB")

    if ERROR_ANALYSIS_ONLY:
        ea = error_analysis_from_cache(stored_results, mode=MODE_TO_RUN)
        print("\n\nERROR ANALYSIS (FROM CACHE ONLY)")
        print(json.dumps(ea, indent=2))
        raise SystemExit(0)

    #fit TF-IDF once on full dataset
    _TFIDF_VECTORIZER = build_tfidf_vectorizer([str(s.get("input", "") or "") for s in cleaned_healthcaremagic])
    print(f"[TFIDF] fit corpus size: {len(cleaned_healthcaremagic)}")

    #preload heavy models
    if EVAL_BERT:
        _get_bert_model()
    if EVAL_NLI:
        _get_nli()

    
    try:
        if EVAL_BERT:
            _ = _bert_embed("warmup")
        if EVAL_NLI:
            _ = nli_proxy_label("warmup premise", "warmup hypothesis")
        try:
            _ = timed_llm_invoke("warmup[influential]", influential_chain, {"input": "warmup question"})
        except Exception:
            pass
    except Exception:
        pass

    results = evaluate_dataset(cleaned_healthcaremagic, max_items=MAX_ITEMS, mode=MODE_TO_RUN)

    print("\n\n" + "="*100)
    print("===== FINAL AGGREGATED RESULTS (ALL BUCKETS/METRICS) =====")
    print(json.dumps(results, indent=2))

    print("\n\n" + "="*100)
    print("ERROR ANALYSIS SUMMARY (ONLINE)")
    print(json.dumps(results["meta"]["error_analysis"], indent=2))

# ---------------------------
# PRINT/LOG CONTROLS
# ---------------------------
LOG_LEVEL = "ERROR"            
PRINT_EVERY = 1                
PRINT_CF_TEXT = True           
PRINT_MAX_CF_CHARS = 2000      
PRINT_TRUNC_INPUT_CHARS = 800
PRINT_TRUNC_OUTPUT_CHARS = 800


PRINT_TIMERS = False           
PRINT_EVAL_START = False       
PRINT_LOAD_LINES = True        


#SPEED/STABILITY (Mac)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NLTK_DATA", "/Users/agapikyrimi/nltk_data")

warnings.filterwarnings("ignore")


#thread tuning
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except Exception:
    pass


# OPTIONAL
SEED = 1234
try:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except Exception:
    pass


#CONFIG
MODE_TO_RUN = "ANT_ONLY"        
MAX_RETRIES = 1               # retries for CF generation
MAX_ITEMS = 100               

CACHE_FLUSH_EVERY = 100
INFLUENTIAL_TOPK_MAX = 2  

#metrics
EVAL_BERT = True
EVAL_NLI = True
EVAL_METEOR = True
EVAL_ROUGE = True

BERT_MODEL_PATH = "/Users/agapikyrimi/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
LLM_MODEL_NAME = "llama2"
LLM_TEMPERATURE = 0.0

NLI_MODEL_NAME = "facebook/bart-large-mnli"
NLI_MAX_LEN = 512

TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 1
TFIDF_MAX_FEATURES = 100000


LLM_TIMEOUT_SEC = 90

EMB_CACHE_MAX = 10_000
TFIDF_VEC_CACHE_MAX = 2_000
NLI_LRU_MAX = 10_000
LLM_LRU_MAX = 10_000


OLLAMA_KEEP_ALIVE: Optional[str] = "30m"


ERROR_ANALYSIS_ONLY = False

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()


#versions/repro metadata
def get_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        versions["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        versions["torch"] = "unknown"
    try:
        import transformers as _tf
        versions["transformers"] = getattr(_tf, "__version__", "unknown")
    except Exception:
        versions["transformers"] = "unknown"
    try:
        import sentence_transformers as _st
        versions["sentence_transformers"] = getattr(_st, "__version__", "unknown")
    except Exception:
        versions["sentence_transformers"] = "unknown"
    return versions


#files/logging
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath(os.getcwd())
os.makedirs(SCRIPT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(
    SCRIPT_DIR,
    f"stored_results_expl_{MODE_TO_RUN}_bert{int(EVAL_BERT)}_nli{int(EVAL_NLI)}_met{int(EVAL_METEOR)}_rouge{int(EVAL_ROUGE)}.pkl"
)
RUNLOG_FILE = os.path.join(SCRIPT_DIR, f"runlog_expl_{MODE_TO_RUN}.txt")

handlers = [logging.StreamHandler()]
try:
    handlers.append(logging.FileHandler(RUNLOG_FILE, mode="a", encoding="utf-8"))
except Exception:
    handlers = [logging.StreamHandler()]

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.ERROR),
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=handlers
)

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            stored_results = pickle.load(f)
    except Exception:
        stored_results = {}
else:
    stored_results = {}


#RSS memory logging
def rss_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None

def _shorten(s: str, n: Optional[int]) -> str:
    s = str(s or "")
    if n is None:
        return s
    return s if len(s) <= n else s[:n] + "...[TRUNCATED]"

def _print_sample(res: Dict[str, Any], idx1: int, total: int, h: str):
    print("\n" + "=" * 100)
    print(f"SAMPLE {idx1}/{total} | hash={h}")
    print("-" * 100)
    print("ORIGINAL INPUT:\n", _shorten(res.get("original_input", ""), PRINT_TRUNC_INPUT_CHARS))
    print("\nORIGINAL OUTPUT:\n", _shorten(res.get("original_output", ""), PRINT_TRUNC_OUTPUT_CHARS))

    print("\nINFLUENTIAL WORDS:", res.get("term_candidates_influential", []))
    print("TF-IDF RAW:", res.get("tfidf_weights_raw", {}))
    print("TF-IDF NORM:", res.get("tfidf_weights_norm", {}))

    cfr = res.get("counterfactuals_robust", []) or []
    cfs = res.get("counterfactuals_strict", []) or []

    print(f"\nCOUNTERFACTUALS ROBUST: {len(cfr)}")
    for j, cf in enumerate(cfr, start=1):
        print("\n" + "-" * 60)
        print(f"[ROBUST #{j}] type={cf.get('type')} status={cf.get('status')} strict={cf.get('strict_valid')}")
        print("term:", cf.get("term"))
        print("replacement:", cf.get("replacement"))
        print("ant_subtype:", cf.get("ant_subtype"))
        print("ungrammatical_proxy:", cf.get("ungrammatical_proxy"))
        print("tfidf_weight_raw:", cf.get("tfidf_weight_raw"), "tfidf_weight_norm:", cf.get("tfidf_weight_norm"))
        print("modified_input:\n", cf.get("modified_input"))
        print("final_advice:\n", cf.get("final_advice"))
        print("nli_proxy:", cf.get("nli_proxy"))
        print("impact_1_minus_bert:", cf.get("impact_1_minus_bert"))
        if PRINT_CF_TEXT:
            txt = cf.get("text", "") or ""
            print("\ncf_text:\n", _shorten(txt, PRINT_MAX_CF_CHARS))

    print(f"\nCOUNTERFACTUALS STRICT: {len(cfs)}")
    for j, cf in enumerate(cfs, start=1):
        print("\n" + "-" * 60)
        print(f"[STRICT #{j}] type={cf.get('type')}")
        print("term:", cf.get("term"))
        print("replacement:", cf.get("replacement"))
        print("ant_subtype:", cf.get("ant_subtype"))
        print("tfidf_weight_raw:", cf.get("tfidf_weight_raw"), "tfidf_weight_norm:", cf.get("tfidf_weight_norm"))
        print("modified_input:\n", cf.get("modified_input"))
        print("final_advice:\n", cf.get("final_advice"))
        print("nli_proxy:", cf.get("nli_proxy"))
        print("impact_1_minus_bert:", cf.get("impact_1_minus_bert"))


#safe timeout for blocking calls (Ollama)
class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError()

def call_with_timeout(fn, seconds: int, *args, **kwargs):
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# Bounded LRU cache
class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = int(maxsize)
        self._d: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, k: str):
        if k not in self._d:
            return None
        self._d.move_to_end(k)
        return self._d[k]

    def set(self, k: str, v: Any):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)



# LLM (Ollama)
def _build_llm() -> OllamaLLM:
    try:
        if OLLAMA_KEEP_ALIVE:
            return OllamaLLM(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, keep_alive=OLLAMA_KEEP_ALIVE)
    except Exception:
        pass
    return OllamaLLM(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

llm = _build_llm()

def timed_llm_invoke(name: str, chain: LLMChain, payload: Dict[str, Any]) -> Any:
    t0 = time.time()
    out = call_with_timeout(chain.invoke, LLM_TIMEOUT_SEC, payload)
    if PRINT_TIMERS:
        print(f"[TIMER][LLM] {name} took {time.time() - t0:.2f}s")
    return out


#helpers functions
def _get_invoke_text(resp: Any, preferred_key: Optional[str] = None) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        if preferred_key and preferred_key in resp and resp[preferred_key] is not None:
            return str(resp[preferred_key])
        for k in ["text", "output", "result", "content"]:
            if k in resp and resp[k] is not None:
                return str(resp[k])
        return str(resp)
    return str(resp)

def safe_parse_json(raw: Any, key: str, default: str) -> str:
    try:
        if isinstance(raw, dict) and key in raw:
            v = str(raw.get(key, "")).strip()
            return v if v else default
        s = str(raw or "")
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            data = json.loads(m.group())
            v = data.get(key, "")
            if v and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    return default

def _norm_phrase(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

#soft-match normalization (punctuation tolerant)

def _normalize_for_soft_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

#tighter boundaries (avoid \w / underscore quirks)

def _phrase_regex(phrase: str) -> str:
    toks = [re.escape(t) for t in (phrase or "").strip().split()]
    if not toks:
        return ""
    joined = r"\s+".join(toks)
    return rf"(?i)(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])"

#exact-regex OR soft-match fallback
    
def term_occurs_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    pat = _phrase_regex(term)
    if pat and re.search(pat, text):
        return True
    return _normalize_for_soft_match(term) in _normalize_for_soft_match(text)

def replace_phrase_in_text(text: str, phrase: str, replacement: str) -> Tuple[str, bool]:
    if not text or not phrase:
        return text, False
    pat = _phrase_regex(phrase)
    if pat and re.search(pat, text) is not None:
        new_text = re.sub(pat, replacement, text, count=1)
        return new_text, new_text != text

    #fallback:soft-match replacement
    src = _normalize_for_soft_match(text)
    ph  = _normalize_for_soft_match(phrase)
    if not ph or ph not in src:
        return text, False

    toks = [re.escape(t) for t in phrase.strip().split()]
    if not toks:
        return text, False
    loose = r"(?i)(?<![A-Za-z0-9])" + r"[^\w]*\s*".join(toks) + r"(?![A-Za-z0-9])"
    if re.search(loose, text) is None:
        return text, False
    new_text = re.sub(loose, replacement, text, count=1)
    return new_text, new_text != text

def ungrammatical_proxy(modified_input: str) -> bool:
    if not modified_input:
        return False
    t = modified_input.lower()
    if "  " in modified_input:
        return True
    if re.search(r"\b(and|or)\s*[,.]", t):
        return True
    if re.search(r"\bis\s+no\s+\w+", t):
        return True
    return False

#Gentler sanitizer:remove trailing explanations only if pattern looks like "TERM definition"/keep slashes as-is (NOT forcing removal)/allow up to 6 words (less truncation)
def sanitize_influential_term(term: str) -> str:
  
    t = (term or "").strip()

    #strip definition-like tail
    for sep in [" - ", " – ", " — ", ": "]:
        if sep in t and len(t.split(sep, 1)[0].split()) <= 6:
            t = t.split(sep, 1)[0].strip()
            break

    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > 6:
        t = " ".join(words[:6]).strip()
    return t




# TF-IDF global&cosine&weights
_TFIDF_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF_VEC_CACHE = LRUCache(maxsize=TFIDF_VEC_CACHE_MAX)

def build_tfidf_vectorizer(corpus_texts: List[str]) -> TfidfVectorizer:
    vect = TfidfVectorizer(
        lowercase=True,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        token_pattern=r"(?u)\b\w+\b",
    )
    vect.fit(corpus_texts if corpus_texts else [""])
    return vect

def _tfidf_vec(text: str):
    global _TFIDF_VECTORIZER
    key = str(text or "")
    v = _TFIDF_VEC_CACHE.get(key)
    if v is None:
        v = _TFIDF_VECTORIZER.transform([key])
        _TFIDF_VEC_CACHE.set(key, v)
    return v

def tfidf_cosine(a: str, b: str) -> float:
    global _TFIDF_VECTORIZER
    if _TFIDF_VECTORIZER is None:
        return 0.0
    va = _tfidf_vec(a or "")
    vb = _tfidf_vec(b or "")
    if va.nnz == 0 and vb.nnz == 0:
        return 0.0
    return float(sk_cosine_similarity(va, vb)[0, 0])

def tfidf_weights_for_terms(question_text: str, terms: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    global _TFIDF_VECTORIZER
    if _TFIDF_VECTORIZER is None or not terms:
        raw0 = {t: 0.0 for t in (terms or [])}
        return raw0, raw0

    vec = _TFIDF_VECTORIZER.transform([question_text or ""])
    if vec.nnz == 0:
        raw0 = {t: 0.0 for t in terms}
        return raw0, raw0

    feature_names = _TFIDF_VECTORIZER.get_feature_names_out()
    feat2idx = {feature_names[i]: i for i in range(len(feature_names))}
    row = vec.tocsr()[0]
    idx2val = {int(i): float(v) for i, v in zip(row.indices, row.data)}

    raw: Dict[str, float] = {}
    for t in terms:
        tn = _norm_phrase(t)
        if not tn:
            raw[t] = 0.0
            continue
        if tn in feat2idx:
            raw[t] = float(idx2val.get(int(feat2idx[tn]), 0.0))
        else:
            best = 0.0
            for tok in tn.split():
                if tok in feat2idx:
                    best = max(best, float(idx2val.get(int(feat2idx[tok]), 0.0)))
            raw[t] = float(best)

    s = float(sum(raw.values())) or 1.0
    norm = {k: float(v) / s for k, v in raw.items()}
    return raw, norm




#PROMPTS
#influentials must be EXACT substrings from QUESTION
influential_prompt = PromptTemplate.from_template(r"""
Return ONLY a valid Python list of exactly 2 strings. No extra text.

Rules (must follow):
- Each string MUST be an EXACT substring copied from the QUESTION (verbatim).
- 1 to 4 words per string.
- Do NOT add explanations/definitions.
- Do NOT add slashes like "A/B".
- Prefer symptoms/conditions/medications that strongly affect advice.

QUESTION:
{input}
""")

synonym_prompt = PromptTemplate.from_template(r"""
You are given ONE influential medical term: "{word}".

Task:
- Generate ONE medically valid synonym (common alternative term).
- If no valid synonym exists, return "no change".

Return ONLY valid JSON:
{{
  "synonym": "..."
}}
""")

antonym_prompt = PromptTemplate.from_template(r"""
You are given ONE influential medical term: "{word}".

Task:
- Return ONE medically plausible OPPOSITE / ANTONYM if it exists as a short phrase.
- If no true antonym exists, return "NO_ANTONYM".

Return ONLY valid JSON:
{{
  "antonym": "..."
}}
""")

negation_prompt = PromptTemplate.from_template(r"""
You are given ONE influential medical term: "{word}".

Task:
- Produce ONE medically plausible NEGATION of the term (absence/denial).
- Output should be SHORT (1–5 words) suitable to replace the original inside a sentence.

Examples:
- "fever" -> "no fever"
- "chest pain" -> "no chest pain"
- "metformin" -> "not taking metformin"

Return ONLY valid JSON:
{{
  "negation": "..."
}}
""")

counterfactual_body_prompt = PromptTemplate.from_template(r"""
You are a careful medical assistant.

Output MUST contain ONLY these 2 sections in order (no extra text):

Modified Input:
"..."

Final Advice:
"..."

Constraints:
- Copy the Modified Input EXACTLY as provided (verbatim inside quotes).
- Final Advice MUST be concrete and in quotes (1–3 sentences).
- Do NOT include reasoning.

Modified Input:
{modified_input}
""")

influential_chain = LLMChain(llm=llm, prompt=influential_prompt, output_key="influential_words")
synonym_chain = LLMChain(llm=llm, prompt=synonym_prompt, output_key="synonym")
antonym_chain = LLMChain(llm=llm, prompt=antonym_prompt, output_key="antonym")
negation_chain = LLMChain(llm=llm, prompt=negation_prompt, output_key="negation")
counterfactual_chain = LLMChain(llm=llm, prompt=counterfactual_body_prompt, output_key="counterfactual")




#signature/hash
def make_prompt_signature() -> str:
    parts = {
        "MODE_TO_RUN": MODE_TO_RUN,
        "LLM_MODEL_NAME": LLM_MODEL_NAME,
        "LLM_TEMPERATURE": str(LLM_TEMPERATURE),
        "MAX_RETRIES": str(MAX_RETRIES),
        "INFLUENTIAL": influential_prompt.template,
        "SYN": synonym_prompt.template,
        "ANT": antonym_prompt.template,
        "NEG": negation_prompt.template,
        "CF": counterfactual_body_prompt.template,
        "TFIDF": f"{TFIDF_NGRAM_RANGE}|{TFIDF_MIN_DF}|{TFIDF_MAX_FEATURES}",
        "EVAL_BERT": str(EVAL_BERT),
        "EVAL_NLI": str(EVAL_NLI),
        "EVAL_ROUGE": str(EVAL_ROUGE),
        "EVAL_METEOR": str(EVAL_METEOR),
        "SEED": str(SEED),
        "DEVICE": str(DEVICE),
    }
    blob = "\n".join([f"{k}={parts[k]}" for k in sorted(parts.keys())])
    return hashlib.md5(blob.encode("utf-8")).hexdigest()

PROMPT_SIG = make_prompt_signature()

def compute_sample_hash(sample: Dict[str, str]) -> str:
    combined = (
        f"{MODE_TO_RUN}||{LLM_MODEL_NAME}||{LLM_TEMPERATURE}||{PROMPT_SIG}||"
        f"{sample.get('instruction','')}||{sample.get('input','')}||{sample.get('output','')}"
    )
    return hashlib.md5(combined.encode("utf-8")).hexdigest()



#influential parsing
def _parse_python_list_of_strings(raw: str) -> List[str]:
    s = (raw or "").strip()
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return []
    items = re.findall(r"""['"]([^'"]+)['"]""", m.group(0))
    out, seen = [], set()
    for it in items:
        itn = _norm_phrase(it)
        if not itn or itn in seen:
            continue
        seen.add(itn)
        out.append(it.strip())
    return out

def _parse_numbered_list(raw: str) -> List[str]:
    s = str(raw or "")
    out, seen = [], set()
    for line in s.splitlines():
        m = re.match(r"^\s*\d+\s*[\.\)]\s*(.+?)\s*$", line)
        if m:
            t = m.group(1).strip()
            tn = _norm_phrase(t)
            if t and tn not in seen:
                seen.add(tn)
                out.append(t)
    return out

def extract_influentials_robust(raw: str) -> List[str]:
    terms = _parse_python_list_of_strings(raw)
    if terms:
        return terms[:INFLUENTIAL_TOPK_MAX]
    terms2 = _parse_numbered_list(raw)
    if terms2:
        return terms2[:INFLUENTIAL_TOPK_MAX]
    return []


#strict validation for CF output
_PLACEHOLDER_MARKERS = {"...", "…", "tbd", "todo", "your final advice here", "final advice here"}

def _is_bad_final_advice(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    tn = _norm_phrase(t)
    if tn in _PLACEHOLDER_MARKERS:
        return True
    if re.fullmatch(r"[.\u2026\-\s]+", t):
        return True
    if len(t) < 12:
        return True
    if re.search(r"[A-Za-z]", t) is None:
        return True
    return False

def extract_final_advice(cf_text: str) -> str:
    if not cf_text:
        return ""
    m = re.search(r'(?is)\bfinal\s+advice\s*:\s*"(.*?)"\s*(?:\n|$)', cf_text)
    if m:
        fa = (m.group(1) or "").strip()
        return "" if _is_bad_final_advice(fa) else fa
    return ""

def is_strict_counterfactual(cf_text: str) -> bool:
    if not cf_text:
        return False
    has_mod = re.search(r'(?is)\bmodified\s+input\s*:\s*"', cf_text) is not None
    has_adv = re.search(r'(?is)\bfinal\s+advice\s*:\s*"', cf_text) is not None
    if not (has_mod and has_adv):
        return False
    return bool(extract_final_advice(cf_text))

def _inject_header(term: str, replacement: str, cf_body: str) -> str:
    return f'--- Counterfactual for "{term}" (Replacement: "{replacement}") ---\n' + (cf_body or "").lstrip()



# Metrics
rouge_scorer_fn = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_BERT_MODEL = None
_EMB_CACHE = LRUCache(maxsize=EMB_CACHE_MAX)

def _get_bert_model():
    global _BERT_MODEL
    if _BERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        if PRINT_LOAD_LINES:
            print(f"[LOAD] SentenceTransformer: {BERT_MODEL_PATH} on {DEVICE}")
        _BERT_MODEL = SentenceTransformer(BERT_MODEL_PATH, device=DEVICE)
    return _BERT_MODEL

def _bert_embed(text: str) -> torch.Tensor:
    key = str(text or "")
    v = _EMB_CACHE.get(key)
    if v is not None:
        return v
    model = _get_bert_model()
    with torch.inference_mode():
        emb = model.encode([key], convert_to_tensor=True, show_progress_bar=False)[0]
    _EMB_CACHE.set(key, emb)
    return emb

def cosine_bert_from_emb(ea: torch.Tensor, eb: torch.Tensor) -> float:
    try:
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))
    except Exception:
        return 0.0

def effect_size_from_emb(ea: torch.Tensor, eb: torch.Tensor) -> float:
    c = cosine_bert_from_emb(ea, eb)
    return float(min(1.0, max(0.0, 1.0 - c)))

def rouge_l(a: str, b: str) -> float:
    if not EVAL_ROUGE:
        return 0.0
    try:
        return rouge_scorer_fn.score(str(a or ""), str(b or ""))["rougeL"].fmeasure
    except Exception:
        return 0.0

_METEOR_FAILURES = 0
def meteor(a: str, b: str) -> float:
    global _METEOR_FAILURES
    if not EVAL_METEOR:
        return 0.0
    aa = str(a or "").strip()
    bb = str(b or "").strip()
    if not aa or not bb:
        return 0.0
    try:
        return meteor_score([bb.split()], aa.split())
    except Exception:
        _METEOR_FAILURES += 1
        return 0.0



#NLI proxy&cache
_NLI_TOKENIZER = None
_NLI_MODEL = None
_ID2LABEL: Optional[Dict[int, str]] = None

def _get_nli():
    global _NLI_TOKENIZER, _NLI_MODEL, _ID2LABEL
    if not EVAL_NLI:
        return None, None, None
    if _NLI_TOKENIZER is None or _NLI_MODEL is None or _ID2LABEL is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if PRINT_LOAD_LINES:
            print(f"[LOAD] NLI model: {NLI_MODEL_NAME} on {DEVICE}")
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE)
        _NLI_MODEL.eval()
        _ID2LABEL = {int(k): str(v).upper() for k, v in _NLI_MODEL.config.id2label.items()}
    return _NLI_TOKENIZER, _NLI_MODEL, _ID2LABEL

def nli_proxy_label(premise: str, hypothesis: str) -> str:
    if not EVAL_NLI:
        return "DISABLED"
    try:
        tok, model, id2label = _get_nli()
        if tok is None or model is None or id2label is None:
            return "DISABLED"
        enc = tok(str(premise or ""), str(hypothesis or ""), return_tensors="pt",
                  truncation=True, max_length=NLI_MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.inference_mode():
            pred_id = int(torch.argmax(model(**enc).logits[0]).item())
        lbl = id2label.get(pred_id, "UNKNOWN")
        if "ENTAIL" in lbl:
            return "ENTAILMENT"
        if "CONTRAD" in lbl:
            return "CONTRADICTION"
        if "NEUT" in lbl:
            return "NEUTRAL"
        return lbl
    except Exception:
        return "UNKNOWN"

@lru_cache(maxsize=NLI_LRU_MAX)
def nli_proxy_label_cached(premise: str, hypothesis: str) -> str:
    return nli_proxy_label(premise, hypothesis)




#cached LLM calls
@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_influentials(question: str) -> Tuple[str, ...]:
    try:
        resp = timed_llm_invoke("influential", influential_chain, {"input": str(question or "")})
    except _TimeoutError:
        return tuple()
    raw = _get_invoke_text(resp, preferred_key="influential_words")
    terms = extract_influentials_robust(raw)
    return tuple(terms[:INFLUENTIAL_TOPK_MAX])

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_synonym(term: str) -> str:
    try:
        resp = timed_llm_invoke("synonym", synonym_chain, {"word": str(term or "")})
    except _TimeoutError:
        return "no change"
    raw = _get_invoke_text(resp, preferred_key="synonym")
    return safe_parse_json(raw, "synonym", "no change")

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_antonym(term: str) -> str:
    try:
        resp = timed_llm_invoke("antonym", antonym_chain, {"word": str(term or "")})
    except _TimeoutError:
        return "NO_ANTONYM"
    raw = _get_invoke_text(resp, preferred_key="antonym")
    return safe_parse_json(raw, "antonym", "NO_ANTONYM")

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_negation(term: str) -> str:
    try:
        resp = timed_llm_invoke("negation", negation_chain, {"word": str(term or "")})
    except _TimeoutError:
        return f"no {term}"
    raw = _get_invoke_text(resp, preferred_key="negation")
    return safe_parse_json(raw, "negation", f"no {term}")

def _synonym_is_usable(term: str, syn: str) -> bool:
    sn = _norm_phrase(syn)
    if not sn or sn == "no change":
        return False
    if sn == _norm_phrase(term):
        return False
    if re.search(r"[.!?]", syn or ""):
        return False
    return True

def _antonym_is_usable(term: str, ant: str) -> bool:
    an = _norm_phrase(ant)
    if not an:
        return False
    if an in {"no_antonym", "no antonym", "no antonym exists", "no_antonm"}:
        return False
    if an == _norm_phrase(term):
        return False
    return True


def choose_ant_or_neg(term: str) -> Tuple[str, str]:
    ant = _cached_antonym(term)
    if _antonym_is_usable(term, ant):
        return ant, "antonym"
    neg = _cached_negation(term)
    neg = neg.strip() if isinstance(neg, str) else ""
    return (neg if neg else f"no {term}"), "negation"

def types_for_mode(mode: str) -> List[str]:
    m = (mode or "").strip().upper()
    if m == "SYN_ONLY":
        return ["syn"]
    if m == "ANT_ONLY":
        return ["ant"]
    return ["syn", "ant"]




#METEOR
def nltk_preflight():
    if not EVAL_METEOR:
        return
    import nltk
    from nltk.data import find

    if "/Users/agapikyrimi/nltk_data" not in nltk.data.path:
        nltk.data.path.insert(0, "/Users/agapikyrimi/nltk_data")

    def _exists_any(candidates: List[str]) -> bool:
        for c in candidates:
            try:
                find(c)
                return True
            except LookupError:
                continue
        return False

    ok_wordnet = _exists_any(["corpora/wordnet", "corpora/wordnet.zip"])
    ok_omw = _exists_any(["corpora/omw-1.4", "corpora/omw-1.4.zip"])

    if not (ok_wordnet and ok_omw):
        print(
            "[NLTK] wordnet missing. METEOR may be slow/fail.\n"
            "Run once:\n"
            "  python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        )




#core: process one sample
        
def process_sample(sample: Dict[str, str], mode: str = MODE_TO_RUN) -> Dict[str, Any]:
    orig_in = str(sample.get("input", "") or "")
    orig_out = str(sample.get("output", "") or "")

    #influentials
    terms_raw = list(_cached_influentials(orig_in))

    #sanitize&keep only those that really occur in text
    terms: List[str] = []
    for t in terms_raw:
        ts = sanitize_influential_term(t)
        if ts and term_occurs_in_text(ts, orig_in):
            terms.append(ts)
        elif t and term_occurs_in_text(t, orig_in):
            terms.append(t)

    
    terms = terms[:INFLUENTIAL_TOPK_MAX]

    w_raw, w_norm = tfidf_weights_for_terms(orig_in, terms)

    results: Dict[str, Any] = {
        "original_input": orig_in,
        "original_output": orig_out,
        "term_candidates_influential": terms,
        "tfidf_weights_raw": w_raw,
        "tfidf_weights_norm": w_norm,
        "counterfactuals_robust": [],
        "counterfactuals_strict": [],
        "per_type_status": {},
    }

    for cf_type in types_for_mode(mode):
        robust = {
            "type": cf_type,
            "term": None,
            "replacement": None,
            "modified_input": None,
            "text": "",
            "final_advice": "",
            "strict_valid": False,
            "impact_1_minus_bert": None,
            "nli_proxy": "UNKNOWN",
            "status": "NOT_ATTEMPTED",
            "ungrammatical_proxy": False,
            "tfidf_weight_raw": 0.0,
            "tfidf_weight_norm": 0.0,
            "ant_subtype": None,
        }

        chosen_term = None
        replacement = None
        modified_input = None
        ant_subtype = None

        for term in terms:
            #never call synonym/antonym if term not in input
            if not term_occurs_in_text(term, orig_in):
                continue

            if cf_type == "syn":
                syn = _cached_synonym(term)
                if not _synonym_is_usable(term, syn):
                    continue
                new_in, did = replace_phrase_in_text(orig_in, term, syn)
                if not did:
                    continue
                chosen_term, replacement, modified_input = term, syn, new_in
                break

            if cf_type == "ant":
                rep, sub = choose_ant_or_neg(term)
                new_in, did = replace_phrase_in_text(orig_in, term, rep)
                if not did:
                    continue
                chosen_term, replacement, modified_input = term, rep, new_in
                ant_subtype = sub
                break

        if chosen_term is None:
            robust["status"] = "NO_MATCH" if terms else "NO_INFLUENTIAL_TERMS"
            results["counterfactuals_robust"].append(robust)
            results["per_type_status"][cf_type] = robust["status"]
            continue

        robust["ungrammatical_proxy"] = bool(ungrammatical_proxy(modified_input))

        cf_body = ""
        for _ in range(max(1, int(MAX_RETRIES))):
            try:
                resp = timed_llm_invoke(
                    f"counterfactual[{cf_type}]",
                    counterfactual_chain,
                    {"modified_input": str(modified_input or "")},
                )
            except _TimeoutError:
                resp = ""
            cf_body = _get_invoke_text(resp, preferred_key="counterfactual") or ""
            if cf_body.strip():
                break

        cf_text = _inject_header(chosen_term, replacement, cf_body)

        robust.update({
            "term": chosen_term,
            "replacement": replacement,
            "modified_input": modified_input,
            "text": cf_text,
            "status": "CF_GENERATED",
            "tfidf_weight_raw": float(w_raw.get(chosen_term, 0.0)),
            "tfidf_weight_norm": float(w_norm.get(chosen_term, 0.0)),
        })
        if cf_type == "ant":
            robust["ant_subtype"] = ant_subtype

        final_advice = str(extract_final_advice(cf_text) or "")
        strict_ok = is_strict_counterfactual(cf_text)

        robust["strict_valid"] = bool(strict_ok)
        robust["final_advice"] = final_advice

        if not final_advice:
            robust["status"] = "NO_FINAL_ADVICE"
            results["counterfactuals_robust"].append(robust)
            results["per_type_status"][cf_type] = robust["status"]
            continue

        robust["nli_proxy"] = str(nli_proxy_label_cached(orig_out, final_advice))

        if strict_ok:
            results["counterfactuals_strict"].append({
                "type": cf_type,
                "term": chosen_term,
                "replacement": replacement,
                "modified_input": modified_input,
                "text": cf_text,
                "final_advice": final_advice,
                "impact_1_minus_bert": None,
                "nli_proxy": robust["nli_proxy"],
                "tfidf_weight_raw": float(w_raw.get(chosen_term, 0.0)),
                "tfidf_weight_norm": float(w_norm.get(chosen_term, 0.0)),
                "ant_subtype": ant_subtype if cf_type == "ant" else None,
            })

        results["counterfactuals_robust"].append(robust)
        results["per_type_status"][cf_type] = robust["status"]

    return results




#error analysis aggregators
def _init_error_counters(types: List[str]):
    return {
        "status_counts": {t: Counter() for t in types},
        "strict_valid": {t: Counter() for t in types},
        "ungrammatical": {t: Counter() for t in types},
        "ant_subtype": Counter(),
        "attempts": Counter(),
    }

def _safe_rate(num: int, den: int) -> Optional[float]:
    return float(num) / float(den) if den else None

def _finalize_error_analysis(counters, types: List[str]) -> Dict[str, Any]:
    counts_per_status_per_type = {t: dict(counters["status_counts"][t]) for t in types}
    strict_valid_rate_per_type = {
        t: _safe_rate(counters["strict_valid"][t]["true"], counters["attempts"][t]) for t in types
    }
    ungrammatical_rate_per_type = {
        t: _safe_rate(counters["ungrammatical"][t]["true"], counters["attempts"][t]) for t in types
    }
    no_match_rate_per_type = {
        t: _safe_rate(
            counters["status_counts"][t]["NO_MATCH"] + counters["status_counts"][t]["NO_INFLUENTIAL_TERMS"],
            counters["attempts"][t]
        )
        for t in types
    }
    no_final_advice_rate_per_type = {
        t: _safe_rate(counters["status_counts"][t]["NO_FINAL_ADVICE"], counters["attempts"][t]) for t in types
    }

    ant_total = sum(counters["ant_subtype"].values())
    ant_subtype_share = {
        "antonym": _safe_rate(counters["ant_subtype"]["antonym"], ant_total),
        "negation": _safe_rate(counters["ant_subtype"]["negation"], ant_total),
        "counts": dict(counters["ant_subtype"]),
    }

    return {
        "counts_per_status_per_type": counts_per_status_per_type,
        "strict_valid_rate_per_type": strict_valid_rate_per_type,
        "ant_subtype_share": ant_subtype_share,
        "ungrammatical_rate_per_type": ungrammatical_rate_per_type,
        "no_match_rate_per_type": no_match_rate_per_type,
        "no_final_advice_rate_per_type": no_final_advice_rate_per_type,
        "attempts_per_type": dict(counters["attempts"]),
    }

def error_analysis_from_cache(stored: Dict[str, Any], mode: str) -> Dict[str, Any]:
    types = types_for_mode(mode)
    counters = _init_error_counters(types)

    for _h, obj in stored.items():
        res = (obj or {}).get("result") or {}
        for cf in res.get("counterfactuals_robust", []) or []:
            typ = (cf.get("type") or "").strip()
            if typ not in types:
                continue
            counters["attempts"][typ] += 1
            st = str(cf.get("status") or "UNKNOWN")
            counters["status_counts"][typ][st] += 1
            counters["strict_valid"][typ]["true" if bool(cf.get("strict_valid")) else "false"] += 1
            counters["ungrammatical"][typ]["true" if bool(cf.get("ungrammatical_proxy")) else "false"] += 1
            if typ == "ant":
                sub = str(cf.get("ant_subtype") or "").strip().lower()
                if sub in {"antonym", "negation"}:
                    counters["ant_subtype"][sub] += 1

    return _finalize_error_analysis(counters, types)




#evaluation
def evaluate_dataset(samples: List[Dict[str, str]], max_items: int = MAX_ITEMS, mode: str = MODE_TO_RUN) -> Dict[str, Any]:
    def new_out_bucket():
        return {"cosine": [], "seq": [], "bert": [], "rougeL": [], "meteor": [], "nli": [], "impact": []}

    def new_in_bucket():
        return {"cosine": [], "seq": [], "bert": [], "rougeL": [], "meteor": []}

    out_keys = ["syn_out_vs_orig_out", "ant_out_vs_orig_out"]
    in_keys  = ["syn_in_vs_orig_in",  "ant_in_vs_orig_in"]

    outR = {k: new_out_bucket() for k in out_keys}
    outS = {k: new_out_bucket() for k in out_keys}
    inR  = {k: new_in_bucket()  for k in in_keys}
    inS  = {k: new_in_bucket()  for k in in_keys}

    types = types_for_mode(mode)
    counters = _init_error_counters(types)

    total = min(len(samples), max_items)

    if PRINT_EVAL_START:
        print("[EVAL] start evaluate_dataset")

    for i, sample in enumerate(samples[:max_items]):
        idx1 = i + 1
        h = compute_sample_hash(sample)

        res = None
        if h in stored_results:
            res = stored_results[h].get("result")

        if res is None:
            try:
                res = process_sample(sample, mode=mode)
            except Exception as e:
             
                err_msg = str(e)

                err_cfs: List[Dict[str, Any]] = []
                for typ in types_for_mode(mode):
                    err_cfs.append({
                        "type": typ,
                        "status": "PROCESS_SAMPLE_EXCEPTION",
                        "error": err_msg,
                        "strict_valid": False,
                        "ungrammatical_proxy": False,
                        "ant_subtype": None,
                        "term": None,
                        "replacement": None,
                        "modified_input": None,
                        "text": "",
                        "final_advice": "",
                        "impact_1_minus_bert": None,
                        "nli_proxy": "UNKNOWN",
                        "tfidf_weight_raw": 0.0,
                        "tfidf_weight_norm": 0.0,
                    })

                stored_results[h] = {
                    "result": {
                        "original_input": str(sample.get("input", "") or ""),
                        "original_output": str(sample.get("output", "") or ""),
                        "term_candidates_influential": [],
                        "tfidf_weights_raw": {},
                        "tfidf_weights_norm": {},
                        "counterfactuals_robust": err_cfs,
                        "counterfactuals_strict": [],
                        "per_type_status": {cf["type"]: "PROCESS_SAMPLE_EXCEPTION" for cf in err_cfs},
                    },
                    "meta": {"prompt_sig": PROMPT_SIG},
                }

                if (idx1) % CACHE_FLUSH_EVERY == 0:
                    try:
                        with open(CACHE_FILE, "wb") as f:
                            pickle.dump(stored_results, f)
                    except Exception:
                        pass
                continue

            stored_results[h] = {"result": res, "meta": {"prompt_sig": PROMPT_SIG}}

       
        if PRINT_EVERY and (idx1 % PRINT_EVERY == 0):
            _print_sample(res, idx1, total, h)

        orig_out = str(res.get("original_output", "") or "")
        orig_in  = str(res.get("original_input", "") or "")

        orig_out_emb = _bert_embed(orig_out) if EVAL_BERT else None
        orig_in_emb  = _bert_embed(orig_in)  if EVAL_BERT else None

        strict_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for scf in res.get("counterfactuals_strict", []) or []:
            k = (str(scf.get("type")), str(scf.get("term")), str(scf.get("replacement")))
            strict_index[k] = scf

        cache_dirty = False

        #counters
        for cf in res.get("counterfactuals_robust", []) or []:
            typ = (cf.get("type") or "").strip()
            if typ not in types:
                continue
            counters["attempts"][typ] += 1
            st = str(cf.get("status") or "UNKNOWN")
            counters["status_counts"][typ][st] += 1
            counters["strict_valid"][typ]["true" if bool(cf.get("strict_valid")) else "false"] += 1
            counters["ungrammatical"][typ]["true" if bool(cf.get("ungrammatical_proxy")) else "false"] += 1
            if typ == "ant":
                sub = str(cf.get("ant_subtype") or "").strip().lower()
                if sub in {"antonym", "negation"}:
                    counters["ant_subtype"][sub] += 1

        #metrics
        for cf in res.get("counterfactuals_robust", []) or []:
            typ = (cf.get("type") or "").strip()
            key_out = f"{typ}_out_vs_orig_out"
            key_in  = f"{typ}_in_vs_orig_in"

            cf_out = str((cf.get("final_advice") or "").strip())
            is_strict = bool(cf.get("strict_valid"))

            if cf_out and key_out in outR:
                if EVAL_BERT and orig_out_emb is not None:
                    cf_out_emb = _bert_embed(cf_out)
                    b = cosine_bert_from_emb(cf_out_emb, orig_out_emb)
                    imp = effect_size_from_emb(cf_out_emb, orig_out_emb)
                else:
                    b, imp = 0.0, None

                if imp is not None and cf.get("impact_1_minus_bert") != float(imp):
                    cf["impact_1_minus_bert"] = float(imp)
                    cache_dirty = True

                outR[key_out]["cosine"].append(tfidf_cosine(cf_out, orig_out))
                outR[key_out]["seq"].append(SequenceMatcher(None, cf_out, orig_out).ratio())
                outR[key_out]["bert"].append(b)
                outR[key_out]["rougeL"].append(rouge_l(cf_out, orig_out))
                outR[key_out]["meteor"].append(meteor(cf_out, orig_out))
                outR[key_out]["nli"].append(cf.get("nli_proxy", "UNKNOWN"))
                if imp is not None:
                    outR[key_out]["impact"].append(float(imp))

                if is_strict and key_out in outS:
                    outS[key_out]["cosine"].append(outR[key_out]["cosine"][-1])
                    outS[key_out]["seq"].append(outR[key_out]["seq"][-1])
                    outS[key_out]["bert"].append(outR[key_out]["bert"][-1])
                    outS[key_out]["rougeL"].append(outR[key_out]["rougeL"][-1])
                    outS[key_out]["meteor"].append(outR[key_out]["meteor"][-1])
                    outS[key_out]["nli"].append(outR[key_out]["nli"][-1])
                    if imp is not None:
                        outS[key_out]["impact"].append(float(imp))

                    k = (str(cf.get("type")), str(cf.get("term")), str(cf.get("replacement")))
                    scf = strict_index.get(k)
                    if scf is not None and imp is not None and scf.get("impact_1_minus_bert") != float(imp):
                        scf["impact_1_minus_bert"] = float(imp)
                        cache_dirty = True

            mod_in = str((cf.get("modified_input") or "").strip())
            if mod_in and key_in in inR:
                if EVAL_BERT and orig_in_emb is not None:
                    mod_in_emb = _bert_embed(mod_in)
                    b_in = cosine_bert_from_emb(mod_in_emb, orig_in_emb)
                else:
                    b_in = 0.0

                inR[key_in]["cosine"].append(tfidf_cosine(mod_in, orig_in))
                inR[key_in]["seq"].append(SequenceMatcher(None, mod_in, orig_in).ratio())
                inR[key_in]["bert"].append(b_in)
                inR[key_in]["rougeL"].append(rouge_l(mod_in, orig_in))
                inR[key_in]["meteor"].append(meteor(mod_in, orig_in))

                if is_strict and key_in in inS:
                    inS[key_in]["cosine"].append(inR[key_in]["cosine"][-1])
                    inS[key_in]["seq"].append(inR[key_in]["seq"][-1])
                    inS[key_in]["bert"].append(inR[key_in]["bert"][-1])
                    inS[key_in]["rougeL"].append(inR[key_in]["rougeL"][-1])
                    inS[key_in]["meteor"].append(inR[key_in]["meteor"][-1])

        if cache_dirty:
            stored_results[h] = {"result": res, "meta": {"prompt_sig": PROMPT_SIG}}

        if (idx1) % CACHE_FLUSH_EVERY == 0:
            try:
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(stored_results, f)
            except Exception:
                pass

    #final
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(stored_results, f)
    except Exception:
        pass

    def mean(x): return float(np.mean(x)) if x else None
    def majority(x): return max(set(x), key=x.count) if x else None

    def pack_out(b):
        return {
            "cosine": mean(b["cosine"]),
            "seq": mean(b["seq"]),
            "bert": mean(b["bert"]),
            "rougeL": mean(b["rougeL"]),
            "meteor": mean(b["meteor"]),
            "nli_proxy_majority": majority(b["nli"]),
            "impact_mean": mean(b["impact"]),
        }

    def pack_in(b):
        return {
            "cosine": mean(b["cosine"]),
            "seq": mean(b["seq"]),
            "bert": mean(b["bert"]),
            "rougeL": mean(b["rougeL"]),
            "meteor": mean(b["meteor"]),
        }

    error_analysis = _finalize_error_analysis(counters, types)

    return {
        "robust_out": {k: pack_out(v) for k, v in outR.items()},
        "strict_out": {k: pack_out(v) for k, v in outS.items()},
        "robust_in":  {k: pack_in(v)  for k, v in inR.items()},
        "strict_in":  {k: pack_in(v)  for k, v in inS.items()},
        "meta": {
            "mode": mode,
            "device": str(DEVICE),
            "seed": int(SEED),
            "cache_file": CACHE_FILE,
            "prompt_sig": PROMPT_SIG,
            "max_items": int(max_items),
            "eval_bert": bool(EVAL_BERT),
            "eval_nli": bool(EVAL_NLI),
            "eval_rouge": bool(EVAL_ROUGE),
            "eval_meteor": bool(EVAL_METEOR),
            "versions": get_versions(),
            "error_analysis": error_analysis,
        },
    }




#main
if __name__ == "__main__":
    nltk_preflight()

    with open("cleaned_healthcaremagic.pkl", "rb") as f:
        cleaned_healthcaremagic = pickle.load(f)

    print(f"[DATA] Loaded dataset with {len(cleaned_healthcaremagic)} samples")
    print(f"[META] seed={SEED} device={DEVICE} versions={get_versions()}")
    print(f"[CACHE] {CACHE_FILE}")
    rb = rss_bytes()
    if rb is not None:
        print(f"[BOOT] RSS={rb/1024/1024:.1f}MB")

    if ERROR_ANALYSIS_ONLY:
        ea = error_analysis_from_cache(stored_results, mode=MODE_TO_RUN)
        print("\n\nERROR ANALYSIS (FROM CACHE ONLY)")
        print(json.dumps(ea, indent=2))
        raise SystemExit(0)

    #fit TF-IDF once on full dataset
    _TFIDF_VECTORIZER = build_tfidf_vectorizer([str(s.get("input", "") or "") for s in cleaned_healthcaremagic])
    print(f"[TFIDF] fit corpus size: {len(cleaned_healthcaremagic)}")

    #preload heavy models
    if EVAL_BERT:
        _get_bert_model()
    if EVAL_NLI:
        _get_nli()

    
    try:
        if EVAL_BERT:
            _ = _bert_embed("warmup")
        if EVAL_NLI:
            _ = nli_proxy_label("warmup premise", "warmup hypothesis")
        try:
            _ = timed_llm_invoke("warmup[influential]", influential_chain, {"input": "warmup question"})
        except Exception:
            pass
    except Exception:
        pass

    results = evaluate_dataset(cleaned_healthcaremagic, max_items=MAX_ITEMS, mode=MODE_TO_RUN)

    print("\n\n" + "="*100)
    print("===== FINAL AGGREGATED RESULTS (ALL BUCKETS/METRICS) =====")
    print(json.dumps(results, indent=2))

    print("\n\n" + "="*100)
    print("ERROR ANALYSIS SUMMARY (ONLINE)")
    print(json.dumps(results["meta"]["error_analysis"], indent=2))


# SYN_ONLY
#2-pass synonym gate (STRICT then RELAXED)

LOG_LEVEL = "ERROR"
PRINT_EVERY = 1
PRINT_CF_TEXT = True
PRINT_MAX_CF_CHARS = 2000
PRINT_TRUNC_INPUT_CHARS = 800
PRINT_TRUNC_OUTPUT_CHARS = 800
PRINT_TIMERS = False
PRINT_EVAL_START = False
PRINT_LOAD_LINES = True


# SPEED/STABILITY (Mac)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NLTK_DATA", "/Users/agapikyrimi/nltk_data")
warnings.filterwarnings("ignore")

try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except Exception:
    pass


#OPTIONAL
SEED = 1234
try:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except Exception:
    pass

# ---------------------------
# CONFIG
# ---------------------------
MODE_TO_RUN = "SYN_ONLY"
RUN_TAG = "synonly_FINAL_scientific_coverage_safe"

MAX_RETRIES = 1
MAX_ITEMS = 100
CACHE_FLUSH_EVERY = 100

INFLUENTIAL_TOPK_MAX = 3

EVAL_BERT = True
EVAL_NLI = True
EVAL_METEOR = True
EVAL_ROUGE = True

BERT_MODEL_PATH = "/Users/agapikyrimi/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
LLM_MODEL_NAME = "llama2"
LLM_TEMPERATURE = 0.0

NLI_MODEL_NAME = "facebook/bart-large-mnli"
NLI_MAX_LEN = 512

TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 1
TFIDF_MAX_FEATURES = 100000

LLM_TIMEOUT_SEC = 90

EMB_CACHE_MAX = 10_000
TFIDF_VEC_CACHE_MAX = 2_000
NLI_LRU_MAX = 10_000
LLM_LRU_MAX = 10_000

OLLAMA_KEEP_ALIVE: Optional[str] = "30m"

ERROR_ANALYSIS_ONLY = False

#cache safety
FORCE_NO_CACHE = False
STRICT_CACHE_PROMPT_SIG = True  #only reuse cache if prompt_sig matches

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()


#Versions / Repro metadata
def get_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        versions["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        versions["torch"] = "unknown"
    try:
        import transformers as _tf
        versions["transformers"] = getattr(_tf, "__version__", "unknown")
    except Exception:
        versions["transformers"] = "unknown"
    try:
        import sentence_transformers as _st
        versions["sentence_transformers"] = getattr(_st, "__version__", "unknown")
    except Exception:
        versions["sentence_transformers"] = "unknown"
    return versions




#files/logging
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath(os.getcwd())
os.makedirs(SCRIPT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(
    SCRIPT_DIR,
    f"stored_results_expl_{MODE_TO_RUN}_{RUN_TAG}_bert{int(EVAL_BERT)}_nli{int(EVAL_NLI)}_met{int(EVAL_METEOR)}_rouge{int(EVAL_ROUGE)}.pkl"
)
RUNLOG_FILE = os.path.join(SCRIPT_DIR, f"runlog_expl_{MODE_TO_RUN}_{RUN_TAG}.txt")

handlers = [logging.StreamHandler()]
try:
    handlers.append(logging.FileHandler(RUNLOG_FILE, mode="a", encoding="utf-8"))
except Exception:
    handlers = [logging.StreamHandler()]

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.ERROR),
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=handlers
)

stored_results: Dict[str, Any] = {}
if not FORCE_NO_CACHE and os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            stored_results = pickle.load(f) or {}
    except Exception:
        stored_results = {}
print(f"[CACHE] file={CACHE_FILE}")
print(f"[CACHE] loaded_entries={len(stored_results)} (FORCE_NO_CACHE={FORCE_NO_CACHE})")


#RSS memory logging
def rss_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None

def _shorten(s: str, n: Optional[int]) -> str:
    s = str(s or "")
    if n is None:
        return s
    return s if len(s) <= n else s[:n] + "...[TRUNCATED]"

def _print_sample(res: Dict[str, Any], idx1: int, total: int, h: str):
    print("\n" + "=" * 100)
    print(f"SAMPLE {idx1}/{total} | hash={h}")
    print("-" * 100)
    print("ORIGINAL INPUT:\n", _shorten(res.get("original_input", ""), PRINT_TRUNC_INPUT_CHARS))
    print("\nORIGINAL OUTPUT:\n", _shorten(res.get("original_output", ""), PRINT_TRUNC_OUTPUT_CHARS))

    print("\nINFLUENTIAL WORDS:", res.get("term_candidates_influential", []))
    print("TF-IDF RAW:", res.get("tfidf_weights_raw", {}))
    print("TF-IDF NORM:", res.get("tfidf_weights_norm", {}))

    cfr = res.get("counterfactuals_robust", []) or []
    cfs = res.get("counterfactuals_strict", []) or []

    print(f"\nCOUNTERFACTUALS ROBUST: {len(cfr)}")
    for j, cf in enumerate(cfr, start=1):
        print("\n" + "-" * 60)
        print(f"[ROBUST #{j}] type={cf.get('type')} status={cf.get('status')} strict={cf.get('strict_valid')}")
        print("term:", cf.get("term"))
        print("replacement:", cf.get("replacement"))
        print("selection_pass:", cf.get("selection_pass"))
        print("ungrammatical_proxy:", cf.get("ungrammatical_proxy"))
        print("tfidf_weight_raw:", cf.get("tfidf_weight_raw"), "tfidf_weight_norm:", cf.get("tfidf_weight_norm"))
        print("modified_input:\n", cf.get("modified_input"))
        print("final_advice:\n", cf.get("final_advice"))
        print("nli_proxy:", cf.get("nli_proxy"))
        print("impact_1_minus_bert:", cf.get("impact_1_minus_bert"))
        if PRINT_CF_TEXT:
            txt = cf.get("text", "") or ""
            print("\ncf_text:\n", _shorten(txt, PRINT_MAX_CF_CHARS))

    print(f"\nCOUNTERFACTUALS STRICT: {len(cfs)}")
    for j, cf in enumerate(cfs, start=1):
        print("\n" + "-" * 60)
        print(f"[STRICT #{j}] type={cf.get('type')}")
        print("term:", cf.get("term"))
        print("replacement:", cf.get("replacement"))
        print("selection_pass:", cf.get("selection_pass"))
        print("modified_input:\n", cf.get("modified_input"))
        print("final_advice:\n", cf.get("final_advice"))
        print("nli_proxy:", cf.get("nli_proxy"))
        print("impact_1_minus_bert:", cf.get("impact_1_minus_bert"))


#safe timeout (Ollama)
class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError()

def _can_use_sigalrm() -> bool:
    if not hasattr(signal, "SIGALRM"):
        return False
    if threading.current_thread() is not threading.main_thread():
        return False
    return True

def call_with_timeout(fn, seconds: int, *args, **kwargs):
    if not _can_use_sigalrm():
        return fn(*args, **kwargs)
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)



class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = int(maxsize)
        self._d: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, k: str):
        if k not in self._d:
            return None
        self._d.move_to_end(k)
        return self._d[k]

    def set(self, k: str, v: Any):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)


#LLM (Ollama)
def _build_llm() -> OllamaLLM:
    try:
        if OLLAMA_KEEP_ALIVE:
            return OllamaLLM(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, keep_alive=OLLAMA_KEEP_ALIVE)
    except Exception:
        pass
    return OllamaLLM(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

llm = _build_llm()

def timed_llm_invoke(name: str, chain: LLMChain, payload: Dict[str, Any]) -> Any:
    t0 = time.time()
    out = call_with_timeout(chain.invoke, LLM_TIMEOUT_SEC, payload)
    if PRINT_TIMERS:
        print(f"[TIMER][LLM] {name} took {time.time() - t0:.2f}s")
    return out

#helpers functions
def _get_invoke_text(resp: Any, preferred_key: Optional[str] = None) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        if preferred_key and preferred_key in resp and resp[preferred_key] is not None:
            return str(resp[preferred_key])
        for k in ["text", "output", "result", "content"]:
            if k in resp and resp[k] is not None:
                return str(resp[k])
        return str(resp)
    return str(resp)

def safe_parse_json(raw: Any, key: str, default: str) -> str:
    try:
        if isinstance(raw, dict) and key in raw:
            v = str(raw.get(key, "")).strip()
            return v if v else default
        s = str(raw or "")
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            data = json.loads(m.group())
            v = data.get(key, "")
            if v and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    return default

def _norm_phrase(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize_for_soft_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _phrase_regex(phrase: str) -> str:
    toks = [re.escape(t) for t in (phrase or "").strip().split()]
    if not toks:
        return ""
    joined = r"\s+".join(toks)
    return rf"(?i)(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])"

def term_occurs_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    pat = _phrase_regex(term)
    if pat and re.search(pat, text):
        return True
    return _normalize_for_soft_match(term) in _normalize_for_soft_match(text)

def replace_phrase_in_text(text: str, phrase: str, replacement: str) -> Tuple[str, bool]:
    if not text or not phrase:
        return text, False
    pat = _phrase_regex(phrase)
    if pat and re.search(pat, text) is not None:
        new_text = re.sub(pat, replacement, text, count=1)
        return new_text, new_text != text

    src = _normalize_for_soft_match(text)
    ph = _normalize_for_soft_match(phrase)
    if not ph or ph not in src:
        return text, False

    toks = [re.escape(t) for t in phrase.strip().split()]
    if not toks:
        return text, False
    loose = r"(?i)(?<![A-Za-z0-9])" + r"[^\w]*\s*".join(toks) + r"(?![A-Za-z0-9])"
    if re.search(loose, text) is None:
        return text, False
    new_text = re.sub(loose, replacement, text, count=1)
    return new_text, new_text != text

def ungrammatical_proxy(modified_input: str) -> bool:
    if not modified_input:
        return False
    t = modified_input.lower()
    if "  " in modified_input:
        return True
    if re.search(r"\b(and|or)\s*[,.]", t):
        return True
    if re.search(r"\bis\s+no\s+\w+", t):
        return True
    return False

def sanitize_influential_term(term: str) -> str:
    t = (term or "").strip()
    for sep in [" - ", " – ", " — ", ": "]:
        if sep in t and len(t.split(sep, 1)[0].split()) <= 6:
            t = t.split(sep, 1)[0].strip()
            break
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > 6:
        t = " ".join(words[:6]).strip()
    return t


# TF-IDF global&cosine&weights
_TFIDF_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF_VEC_CACHE = LRUCache(maxsize=TFIDF_VEC_CACHE_MAX)

def build_tfidf_vectorizer(corpus_texts: List[str]) -> TfidfVectorizer:
    vect = TfidfVectorizer(
        lowercase=True,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        token_pattern=r"(?u)\b\w+\b",
    )
    vect.fit(corpus_texts if corpus_texts else [""])
    return vect

def _tfidf_vec(text: str):
    global _TFIDF_VECTORIZER
    key = str(text or "")
    v = _TFIDF_VEC_CACHE.get(key)
    if v is None:
        v = _TFIDF_VECTORIZER.transform([key])
        _TFIDF_VEC_CACHE.set(key, v)
    return v

def tfidf_cosine(a: str, b: str) -> float:
    global _TFIDF_VECTORIZER
    if _TFIDF_VECTORIZER is None:
        return 0.0
    va = _tfidf_vec(a or "")
    vb = _tfidf_vec(b or "")
    if va.nnz == 0 and vb.nnz == 0:
        return 0.0
    return float(sk_cosine_similarity(va, vb)[0, 0])

def tfidf_weights_for_terms(question_text: str, terms: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    global _TFIDF_VECTORIZER
    if _TFIDF_VECTORIZER is None or not terms:
        raw0 = {t: 0.0 for t in (terms or [])}
        return raw0, raw0

    vec = _TFIDF_VECTORIZER.transform([question_text or ""])
    if vec.nnz == 0:
        raw0 = {t: 0.0 for t in terms}
        return raw0, raw0

    feature_names = _TFIDF_VECTORIZER.get_feature_names_out()
    feat2idx = {feature_names[i]: i for i in range(len(feature_names))}
    row = vec.tocsr()[0]
    idx2val = {int(i): float(v) for i, v in zip(row.indices, row.data)}

    raw: Dict[str, float] = {}
    for t in terms:
        tn = _norm_phrase(t)
        if not tn:
            raw[t] = 0.0
            continue
        if tn in feat2idx:
            raw[t] = float(idx2val.get(int(feat2idx[tn]), 0.0))
        else:
            best = 0.0
            for tok in tn.split():
                if tok in feat2idx:
                    best = max(best, float(idx2val.get(int(feat2idx[tok]), 0.0)))
            raw[t] = float(best)

    s = float(sum(raw.values())) or 1.0
    norm = {k: float(v) / s for k, v in raw.items()}
    return raw, norm

def tfidf_fallback_influentials(question_text: str, topk: int) -> List[str]:
    global _TFIDF_VECTORIZER
    if _TFIDF_VECTORIZER is None:
        return []

    q = str(question_text or "")
    vec = _TFIDF_VECTORIZER.transform([q])
    row = vec.tocsr()[0]
    if row.nnz == 0:
        return []

    feats = _TFIDF_VECTORIZER.get_feature_names_out()
    pairs = sorted(zip(row.indices, row.data), key=lambda x: float(x[1]), reverse=True)

    out: List[str] = []
    seen = set()

    for idx, _w in pairs:
        term = str(feats[int(idx)])
        if len(term.split()) < 1 or len(term.split()) > 4:
            continue
        if not term_occurs_in_text(term, q):
            continue
        key = _norm_phrase(term)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= int(topk):
            break

    return out


# PROMPTS (SYN ONLY)
influential_prompt = PromptTemplate.from_template(r"""
Return ONLY a valid Python list of up to 3 strings. No extra text.

Rules (must follow):
- Each string MUST be an EXACT substring copied from the QUESTION (verbatim).
- 1 to 4 words per string.
- Do NOT add explanations/definitions.
- Prefer symptoms/conditions/medications that strongly affect advice.

QUESTION:
{input}
""")

synonym_prompt_strict = PromptTemplate.from_template(r"""
You are given ONE influential medical term: "{word}".

Return ONE medically valid synonym that is fully interchangeable in the same sentence (same meaning).
- 1–4 words.
- Avoid abbreviations unless original is an abbreviation.
- If not certain it's fully interchangeable, return "no change".

Return ONLY valid JSON:
{{
  "synonym": "..."
}}
""")

synonym_prompt_relaxed = PromptTemplate.from_template(r"""
You are given ONE medical term from a patient's question: "{word}".

Return ONE SHORT common alternative phrasing that preserves meaning in context.
- 1–6 words.
- Keep the same clinical meaning (do not add severity/duration/new symptoms).
- If you cannot provide a safe meaning-preserving alternative, return "no change".

Return ONLY valid JSON:
{{
  "synonym": "..."
}}
""")

counterfactual_body_prompt = PromptTemplate.from_template(r"""
You are a careful medical assistant.

You will be given:
(1) Modified Input (synonym-only change)
(2) Original Output (the original advice)

GOAL (Synonym Robustness):
- Keep the Final Advice semantically equivalent to the Original Output.
- Keep recommendations/cautions/tone the same.
- Make the smallest possible edits.
- If the Original Output still fits the Modified Input, you MAY copy it verbatim.
- Do NOT introduce new diagnoses/tests/medications.
- Do NOT remove safety warnings present in the Original Output.
- Do NOT mention rewriting/synonyms/original.

Output MUST contain ONLY these 2 sections in order (no extra text):

Modified Input:
"..."

Final Advice:
"..."

Constraints:
- Copy the Modified Input EXACTLY as provided (verbatim inside quotes).
- Final Advice MUST be concrete, in quotes, 1–3 sentences.
- Do NOT include reasoning.

Modified Input:
{modified_input}

Original Output:
{original_output}
""")

influential_chain = LLMChain(llm=llm, prompt=influential_prompt, output_key="influential_words")
synonym_chain_strict = LLMChain(llm=llm, prompt=synonym_prompt_strict, output_key="synonym")
synonym_chain_relaxed = LLMChain(llm=llm, prompt=synonym_prompt_relaxed, output_key="synonym")
counterfactual_chain = LLMChain(llm=llm, prompt=counterfactual_body_prompt, output_key="counterfactual")


# signature/hash
def make_prompt_signature() -> str:
    parts = {
        "MODE_TO_RUN": MODE_TO_RUN,
        "RUN_TAG": RUN_TAG,
        "LLM_MODEL_NAME": LLM_MODEL_NAME,
        "LLM_TEMPERATURE": str(LLM_TEMPERATURE),
        "MAX_RETRIES": str(MAX_RETRIES),
        "INFLUENTIAL": influential_prompt.template,
        "SYN_STRICT": synonym_prompt_strict.template,
        "SYN_RELAXED": synonym_prompt_relaxed.template,
        "CF_SYN_MINEDIT": counterfactual_body_prompt.template,
        "TFIDF": f"{TFIDF_NGRAM_RANGE}|{TFIDF_MIN_DF}|{TFIDF_MAX_FEATURES}",
        "EVAL_BERT": str(EVAL_BERT),
        "EVAL_NLI": str(EVAL_NLI),
        "EVAL_ROUGE": str(EVAL_ROUGE),
        "EVAL_METEOR": str(EVAL_METEOR),
        "SEED": str(SEED),
        "DEVICE": str(DEVICE),
    }
    blob = "\n".join([f"{k}={parts[k]}" for k in sorted(parts.keys())])
    return hashlib.md5(blob.encode("utf-8")).hexdigest()

PROMPT_SIG = make_prompt_signature()

def compute_sample_hash(sample: Dict[str, str]) -> str:
    combined = (
        f"{MODE_TO_RUN}||{RUN_TAG}||{LLM_MODEL_NAME}||{LLM_TEMPERATURE}||{PROMPT_SIG}||"
        f"{sample.get('instruction','')}||{sample.get('input','')}||{sample.get('output','')}"
    )
    return hashlib.md5(combined.encode("utf-8")).hexdigest()




#influential parsing
def _parse_python_list_of_strings(raw: str) -> List[str]:
    s = (raw or "").strip()
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return []
    items = re.findall(r"""['"]([^'"]+)['"]""", m.group(0))
    out, seen = [], set()
    for it in items:
        itn = _norm_phrase(it)
        if not itn or itn in seen:
            continue
        seen.add(itn)
        out.append(it.strip())
    return out

def _parse_numbered_list(raw: str) -> List[str]:
    s = str(raw or "")
    out, seen = [], set()
    for line in s.splitlines():
        m = re.match(r"^\s*\d+\s*[\.\)]\s*(.+?)\s*$", line)
        if m:
            t = m.group(1).strip()
            tn = _norm_phrase(t)
            if t and tn not in seen:
                seen.add(tn)
                out.append(t)
    return out

def extract_influentials_robust(raw: str) -> List[str]:
    terms = _parse_python_list_of_strings(raw)
    if terms:
        return terms[:INFLUENTIAL_TOPK_MAX]
    terms2 = _parse_numbered_list(raw)
    if terms2:
        return terms2[:INFLUENTIAL_TOPK_MAX]
    return []



#strict validation for CF output
_PLACEHOLDER_MARKERS = {"...", "…", "tbd", "todo", "your final advice here", "final advice here"}

def _is_bad_final_advice(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    tn = _norm_phrase(t)
    if tn in _PLACEHOLDER_MARKERS:
        return True
    if re.fullmatch(r"[.\u2026\-\s]+", t):
        return True
    if len(t) < 12:
        return True
    if re.search(r"[A-Za-z]", t) is None:
        return True
    return False

#tolerant extractor
def extract_final_advice(cf_text: str) -> str:
    if not cf_text:
        return ""

    m = re.search(r'(?is)\bfinal\s+advice\s*:\s*"(.*?)"\s*(?:\n|$)', cf_text)
    if m:
        fa = (m.group(1) or "").strip()
        return "" if _is_bad_final_advice(fa) else fa

    m2 = re.search(r'(?is)\bfinal\s+advice\s*:\s*(.+?)\s*$', cf_text)
    if m2:
        fa = (m2.group(1) or "").strip()
        fa = fa.strip('"').strip()
        fa = re.split(r'(?is)\bmodified\s+input\s*:\s*', fa)[0].strip()
        fa = re.split(r'(?is)\bfinal\s+advice\s*:\s*', fa)[0].strip()
        return "" if _is_bad_final_advice(fa) else fa

    return ""

#strict validity (keep mod quoted requirement, relax final advice quotes)
def is_strict_counterfactual(cf_text: str) -> bool:
    if not cf_text:
        return False
    has_mod = re.search(r'(?is)\bmodified\s+input\s*:\s*"', cf_text) is not None
    has_adv = re.search(r'(?is)\bfinal\s+advice\s*:\s*', cf_text) is not None
    if not (has_mod and has_adv):
        return False
    return bool(extract_final_advice(cf_text))

def _inject_header(term: str, replacement: str, cf_body: str) -> str:
    return f'--- Counterfactual for "{term}" (Replacement: "{replacement}") ---\n' + (cf_body or "").lstrip()

#postprocess CF body to enforce exact Modified Input quoting
def _normalize_counterfactual_body(modified_input: str, cf_body_raw: str) -> str:
    mi = str(modified_input or "")
    raw = str(cf_body_raw or "")

    fa = extract_final_advice(raw)
    if not fa:
        return raw.strip()

    fa = fa.strip().strip('"').strip()
    if _is_bad_final_advice(fa):
        return raw.strip()

    return (
        'Modified Input:\n'
        f'"{mi}"\n\n'
        'Final Advice:\n'
        f'"{fa}"\n'
    )

#metrics
rouge_scorer_fn = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_BERT_MODEL = None
_EMB_CACHE = LRUCache(maxsize=EMB_CACHE_MAX)

def _get_bert_model():
    global _BERT_MODEL
    if _BERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        if PRINT_LOAD_LINES:
            print(f"[LOAD] SentenceTransformer: {BERT_MODEL_PATH} on {DEVICE}")
        _BERT_MODEL = SentenceTransformer(BERT_MODEL_PATH, device=DEVICE)
    return _BERT_MODEL

def _bert_embed(text: str) -> torch.Tensor:
    key = str(text or "")
    v = _EMB_CACHE.get(key)
    if v is not None:
        return v
    model = _get_bert_model()
    with torch.inference_mode():
        emb = model.encode([key], convert_to_tensor=True, show_progress_bar=False)[0]
    _EMB_CACHE.set(key, emb)
    return emb

def cosine_bert_from_emb(ea: torch.Tensor, eb: torch.Tensor) -> float:
    try:
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))
    except Exception:
        return 0.0

def effect_size_from_emb(ea: torch.Tensor, eb: torch.Tensor) -> float:
    c = cosine_bert_from_emb(ea, eb)
    return float(min(1.0, max(0.0, 1.0 - c)))

def rouge_l(a: str, b: str) -> float:
    if not EVAL_ROUGE:
        return 0.0
    try:
        return rouge_scorer_fn.score(str(a or ""), str(b or ""))["rougeL"].fmeasure
    except Exception:
        return 0.0

_METEOR_FAILURES = 0
def meteor(a: str, b: str) -> float:
    global _METEOR_FAILURES
    if not EVAL_METEOR:
        return 0.0
    aa = str(a or "").strip()
    bb = str(b or "").strip()
    if not aa or not bb:
        return 0.0
    try:
        return meteor_score([bb.split()], aa.split())
    except Exception:
        _METEOR_FAILURES += 1
        return 0.0


#NLI proxy (lazy)&cache
_NLI_TOKENIZER = None
_NLI_MODEL = None
_ID2LABEL: Optional[Dict[int, str]] = None

def _get_nli():
    global _NLI_TOKENIZER, _NLI_MODEL, _ID2LABEL
    if not EVAL_NLI:
        return None, None, None
    if _NLI_TOKENIZER is None or _NLI_MODEL is None or _ID2LABEL is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if PRINT_LOAD_LINES:
            print(f"[LOAD] NLI model: {NLI_MODEL_NAME} on {DEVICE}")
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE)
        _NLI_MODEL.eval()
        _ID2LABEL = {int(k): str(v).upper() for k, v in _NLI_MODEL.config.id2label.items()}
    return _NLI_TOKENIZER, _NLI_MODEL, _ID2LABEL

def nli_proxy_label(premise: str, hypothesis: str) -> str:
    if not EVAL_NLI:
        return "DISABLED"
    try:
        tok, model, id2label = _get_nli()
        if tok is None or model is None or id2label is None:
            return "DISABLED"
        enc = tok(str(premise or ""), str(hypothesis or ""), return_tensors="pt",
                  truncation=True, max_length=NLI_MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.inference_mode():
            pred_id = int(torch.argmax(model(**enc).logits[0]).item())
        lbl = id2label.get(pred_id, "UNKNOWN")
        if "ENTAIL" in lbl:
            return "ENTAILMENT"
        if "CONTRAD" in lbl:
            return "CONTRADICTION"
        if "NEUT" in lbl:
            return "NEUTRAL"
        return lbl
    except Exception:
        return "UNKNOWN"

@lru_cache(maxsize=NLI_LRU_MAX)
def nli_proxy_label_cached(premise: str, hypothesis: str) -> str:
    return nli_proxy_label(premise, hypothesis)




# LLM cached calls

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_influentials(question: str) -> Tuple[str, ...]:
    try:
        resp = timed_llm_invoke("influential", influential_chain, {"input": str(question or "")})
    except _TimeoutError:
        return tuple()
    raw = _get_invoke_text(resp, preferred_key="influential_words")
    terms = extract_influentials_robust(raw)
    return tuple(terms[:INFLUENTIAL_TOPK_MAX])

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_synonym_strict(term: str) -> str:
    try:
        resp = timed_llm_invoke("synonym_strict", synonym_chain_strict, {"word": str(term or "")})
    except _TimeoutError:
        return "no change"
    raw = _get_invoke_text(resp, preferred_key="synonym")
    return safe_parse_json(raw, "synonym", "no change")

@lru_cache(maxsize=LLM_LRU_MAX)
def _cached_synonym_relaxed(term: str) -> str:
    try:
        resp = timed_llm_invoke("synonym_relaxed", synonym_chain_relaxed, {"word": str(term or "")})
    except _TimeoutError:
        return "no change"
    raw = _get_invoke_text(resp, preferred_key="synonym")
    return safe_parse_json(raw, "synonym", "no change")


# Scientific gates (STRICT + RELAXED)
def _synonym_is_usable_basic(term: str, syn: str) -> bool:
    sn = _norm_phrase(syn)
    if not sn or sn == "no change":
        return False
    if sn == _norm_phrase(term):
        return False
    if re.search(r"[.!?]", syn or ""):
        return False
    return True

def _contains_digits(s: str) -> bool:
    return re.search(r"[0-9]", s or "") is not None

def _looks_like_proper_noun_or_drug(term: str) -> bool:
    t = (term or "").strip()
    if len(t.split()) != 1:
        return False
    if not t[:1].isupper():
        return False
    if re.search(r"[^A-Za-z]", t):
        return False
    return True

#clinical-shift block
_RISKY_SYNONYM_WORDS = {
    "tumor", "cancer", "carcinoma", "malignancy", "metastasis",
    "stroke", "infarct", "aneurysm", "sepsis", "meningitis",
    "fracture", "pneumonia", "embolism", "myocardial"
}

def _contains_risky_word(s: str) -> bool:
    t = _norm_phrase(s)
    for w in _RISKY_SYNONYM_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return True
    return False

def _clinical_shift_block(term: str, syn: str) -> bool:
    if _contains_risky_word(syn) and (not _contains_risky_word(term)):
        return True
    return False

#FIX: -ing/POS context gate
def _needs_ing_form(term: str, text: str) -> bool:
    t = (term or "").strip()
    if not t.lower().endswith("ing"):
        return False
    #if term appears after copula/aux verbs in the question, it is likely a gerund/participle slot
    pat = rf"(?i)\b(is|was|are|were|been|be)\s+{re.escape(t)}\b"
    return re.search(pat, text or "") is not None

def _synonym_matches_ing_requirement(term: str, syn: str, orig_in: str) -> bool:
    if not _needs_ing_form(term, orig_in):
        return True
    s = (syn or "").strip().lower()
    # require at least one token ending with -ing (e.g., "rotating", "spinning sensation" also passes because "spinning" ends with -ing)
    return any(tok.endswith("ing") for tok in s.split())

def synonym_gate_strict(term: str, syn: str) -> bool:
    s = (syn or "").strip()
    if not _synonym_is_usable_basic(term, s):
        return False
    if len(s.split()) > 4:
        return False
    if any(ch in s for ch in ["(", ")", "/", ";", ":"]):
        return False
    if _contains_digits(s) and (not _contains_digits(term)):
        return False
    if _clinical_shift_block(term, s):
        return False
    return True

def synonym_gate_relaxed(term: str, syn: str) -> bool:
    s = (syn or "").strip()
    if not _synonym_is_usable_basic(term, s):
        return False
    if len(s.split()) > 6:
        return False
    if any(ch in s for ch in ["{", "}", "[", "]"]):
        return False
    if _contains_digits(s) and (not _contains_digits(term)):
        return False
    if _clinical_shift_block(term, s):
        return False
    return True

def types_for_mode(mode: str) -> List[str]:
    return ["syn"]


# METEOR
def nltk_preflight():
    if not EVAL_METEOR:
        return
    import nltk
    from nltk.data import find
    if "/Users/agapikyrimi/nltk_data" not in nltk.data.path:
        nltk.data.path.insert(0, "/Users/agapikyrimi/nltk_data")

    def _exists_any(candidates: List[str]) -> bool:
        for c in candidates:
            try:
                find(c)
                return True
            except LookupError:
                continue
        return False

    ok_wordnet = _exists_any(["corpora/wordnet", "corpora/wordnet.zip"])
    ok_omw = _exists_any(["corpora/omw-1.4", "corpora/omw-1.4.zip"])

    if not (ok_wordnet and ok_omw):
        print(
            "[NLTK] wordnet missing. METEOR may be slow/fail.\n"
            "Run once:\n"
            "  python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        )


#cache
def make_prompt_signature_only() -> str:
    return PROMPT_SIG

#Core: process one sample (SYN_ONLY)
def process_sample(sample: Dict[str, str], mode: str = MODE_TO_RUN) -> Dict[str, Any]:
    global _TFIDF_VECTORIZER
    if _TFIDF_VECTORIZER is None:
        raise RuntimeError("TFIDF vectorizer is None inside process_sample (fit before evaluate_dataset)")

    orig_in = str(sample.get("input", "") or "")
    orig_out = str(sample.get("output", "") or "")

    terms_raw = list(_cached_influentials(orig_in))
    if not terms_raw:
        terms_raw = tfidf_fallback_influentials(orig_in, topk=INFLUENTIAL_TOPK_MAX)

    terms: List[str] = []
    for t in terms_raw:
        ts = sanitize_influential_term(t)
        if ts and term_occurs_in_text(ts, orig_in):
            terms.append(ts)
        elif t and term_occurs_in_text(t, orig_in):
            terms.append(t)
    terms = terms[:INFLUENTIAL_TOPK_MAX]

    w_raw, w_norm = tfidf_weights_for_terms(orig_in, terms)

    results: Dict[str, Any] = {
        "original_input": orig_in,
        "original_output": orig_out,
        "term_candidates_influential": terms,
        "tfidf_weights_raw": w_raw,
        "tfidf_weights_norm": w_norm,
        "counterfactuals_robust": [],
        "counterfactuals_strict": [],
        "per_type_status": {},
    }

    for cf_type in types_for_mode(mode):
        robust = {
            "type": cf_type,
            "term": None,
            "replacement": None,
            "modified_input": None,
            "text": "",
            "final_advice": "",
            "strict_valid": False,
            "impact_1_minus_bert": None,
            "nli_proxy": "UNKNOWN",
            "status": "NOT_ATTEMPTED",
            "ungrammatical_proxy": False,
            "tfidf_weight_raw": 0.0,
            "tfidf_weight_norm": 0.0,
            "selection_pass": None,  #STRICT or RELAXED
        }

        chosen_term = None
        replacement = None
        modified_input = None
        selection_pass = None

        
        #STRICT
        for term in terms:
            if not term_occurs_in_text(term, orig_in):
                continue
            if _looks_like_proper_noun_or_drug(term):
                continue

            syn = _cached_synonym_strict(term)
            if not synonym_gate_strict(term, syn):
                continue
            # ✅ FINAL POS FIX
            if not _synonym_matches_ing_requirement(term, syn, orig_in):
                continue

            new_in, did = replace_phrase_in_text(orig_in, term, syn)
            if not did:
                continue

            chosen_term, replacement, modified_input = term, syn, new_in
            selection_pass = "STRICT"
            break

       
        #RELAXED fallback
        if chosen_term is None:
            for term in terms:
                if not term_occurs_in_text(term, orig_in):
                    continue
                if _looks_like_proper_noun_or_drug(term):
                    continue

                syn = _cached_synonym_relaxed(term)
                if not synonym_gate_relaxed(term, syn):
                    continue
                # ✅ FINAL POS FIX also in RELAXED (prevents same grammar bug)
                if not _synonym_matches_ing_requirement(term, syn, orig_in):
                    continue

                new_in, did = replace_phrase_in_text(orig_in, term, syn)
                if not did:
                    continue

                chosen_term, replacement, modified_input = term, syn, new_in
                selection_pass = "RELAXED"
                break

        if chosen_term is None:
            robust["status"] = "NO_MATCH" if terms else "NO_INFLUENTIAL_TERMS"
            results["counterfactuals_robust"].append(robust)
            results["per_type_status"][cf_type] = robust["status"]
            continue

        robust["ungrammatical_proxy"] = bool(ungrammatical_proxy(modified_input))

        cf_body = ""
        for _ in range(max(1, int(MAX_RETRIES))):
            try:
                resp = timed_llm_invoke(
                    f"counterfactual[{cf_type}]",
                    counterfactual_chain,
                    {"modified_input": str(modified_input or ""), "original_output": str(orig_out or "")},
                )
            except _TimeoutError:
                resp = ""
            cf_body = _get_invoke_text(resp, preferred_key="counterfactual") or ""
            if cf_body.strip():
                break

        cf_body_norm = _normalize_counterfactual_body(modified_input, cf_body)
        cf_text = _inject_header(chosen_term, replacement, cf_body_norm)

        robust.update({
            "term": chosen_term,
            "replacement": replacement,
            "modified_input": modified_input,
            "text": cf_text,
            "status": "CF_GENERATED",
            "tfidf_weight_raw": float(w_raw.get(chosen_term, 0.0)),
            "tfidf_weight_norm": float(w_norm.get(chosen_term, 0.0)),
            "selection_pass": selection_pass,
        })

        final_advice = str(extract_final_advice(cf_text) or "")
        strict_ok = is_strict_counterfactual(cf_text)

        robust["strict_valid"] = bool(strict_ok)
        robust["final_advice"] = final_advice

        if not final_advice:
            robust["status"] = "NO_FINAL_ADVICE"
            results["counterfactuals_robust"].append(robust)
            results["per_type_status"][cf_type] = robust["status"]
            continue

        robust["nli_proxy"] = str(nli_proxy_label_cached(orig_out, final_advice))

        if strict_ok:
            results["counterfactuals_strict"].append({
                "type": cf_type,
                "term": chosen_term,
                "replacement": replacement,
                "modified_input": modified_input,
                "text": cf_text,
                "final_advice": final_advice,
                "impact_1_minus_bert": None,
                "nli_proxy": robust["nli_proxy"],
                "tfidf_weight_raw": float(w_raw.get(chosen_term, 0.0)),
                "tfidf_weight_norm": float(w_norm.get(chosen_term, 0.0)),
                "selection_pass": selection_pass,
            })

        results["counterfactuals_robust"].append(robust)
        results["per_type_status"][cf_type] = robust["status"]

    return results





#error analysis aggregators
def _init_error_counters(types: List[str]):
    return {
        "status_counts": {t: Counter() for t in types},
        "strict_valid": {t: Counter() for t in types},
        "ungrammatical": {t: Counter() for t in types},
        "attempts": Counter(),
    }

def _safe_rate(num: int, den: int) -> Optional[float]:
    return float(num) / float(den) if den else None

def _finalize_error_analysis(counters, types: List[str]) -> Dict[str, Any]:
    counts_per_status_per_type = {t: dict(counters["status_counts"][t]) for t in types}
    strict_valid_rate_per_type = {
        t: _safe_rate(counters["strict_valid"][t]["true"], counters["attempts"][t]) for t in types
    }
    ungrammatical_rate_per_type = {
        t: _safe_rate(counters["ungrammatical"][t]["true"], counters["attempts"][t]) for t in types
    }
    no_match_rate_per_type = {
        t: _safe_rate(
            counters["status_counts"][t]["NO_MATCH"] + counters["status_counts"][t]["NO_INFLUENTIAL_TERMS"],
            counters["attempts"][t]
        )
        for t in types
    }
    no_final_advice_rate_per_type = {
        t: _safe_rate(counters["status_counts"][t]["NO_FINAL_ADVICE"], counters["attempts"][t]) for t in types
    }

    return {
        "counts_per_status_per_type": counts_per_status_per_type,
        "strict_valid_rate_per_type": strict_valid_rate_per_type,
        "ungrammatical_rate_per_type": ungrammatical_rate_per_type,
        "no_match_rate_per_type": no_match_rate_per_type,
        "no_final_advice_rate_per_type": no_final_advice_rate_per_type,
        "attempts_per_type": dict(counters["attempts"]),
    }

def error_analysis_from_cache(stored: Dict[str, Any], mode: str) -> Dict[str, Any]:
    types = types_for_mode(mode)
    counters = _init_error_counters(types)

    for _h, obj in stored.items():
        res = (obj or {}).get("result") or {}
        for cf in res.get("counterfactuals_robust", []) or []:
            typ = (cf.get("type") or "").strip()
            if typ not in types:
                continue
            counters["attempts"][typ] += 1
            st = str(cf.get("status") or "UNKNOWN")
            counters["status_counts"][typ][st] += 1
            counters["strict_valid"][typ]["true" if bool(cf.get("strict_valid")) else "false"] += 1
            counters["ungrammatical"][typ]["true" if bool(cf.get("ungrammatical_proxy")) else "false"] += 1

    return _finalize_error_analysis(counters, types)





#evaluation
def evaluate_dataset(samples: List[Dict[str, str]], max_items: int = MAX_ITEMS, mode: str = MODE_TO_RUN) -> Dict[str, Any]:
    def new_out_bucket():
        return {"cosine": [], "seq": [], "bert": [], "rougeL": [], "meteor": [], "nli": [], "impact": []}

    def new_in_bucket():
        return {"cosine": [], "seq": [], "bert": [], "rougeL": [], "meteor": []}

    out_keys = ["syn_out_vs_orig_out"]
    in_keys = ["syn_in_vs_orig_in"]

    outR = {k: new_out_bucket() for k in out_keys}
    outS = {k: new_out_bucket() for k in out_keys}
    inR = {k: new_in_bucket() for k in in_keys}
    inS = {k: new_in_bucket() for k in in_keys}

    types = types_for_mode(mode)
    counters = _init_error_counters(types)

    total = min(len(samples), int(max_items))
    print(f"[EVAL] total_to_run={total} (len_dataset={len(samples)} max_items={max_items})")
    if PRINT_EVAL_START:
        print("[EVAL] start evaluate_dataset")

    for i, sample in enumerate(samples[:total]):
        idx1 = i + 1
        print(f"[DEBUG] entering loop idx1={idx1}")
        h = compute_sample_hash(sample)

        res = None
        obj = stored_results.get(h)
        if obj is not None:
            if (not STRICT_CACHE_PROMPT_SIG) or (obj.get("meta", {}).get("prompt_sig") == PROMPT_SIG):
                res = obj.get("result")

        if res is None:
            try:
                res = process_sample(sample, mode=mode)
            except Exception as e:
                print(f"[EXC] idx1={idx1} hash={h} error={type(e).__name__}: {e}")
                err_msg = str(e)
                err_cfs: List[Dict[str, Any]] = [{
                    "type": "syn",
                    "status": "PROCESS_SAMPLE_EXCEPTION",
                    "error": err_msg,
                    "strict_valid": False,
                    "ungrammatical_proxy": False,
                    "term": None,
                    "replacement": None,
                    "modified_input": None,
                    "text": "",
                    "final_advice": "",
                    "impact_1_minus_bert": None,
                    "nli_proxy": "UNKNOWN",
                    "tfidf_weight_raw": 0.0,
                    "tfidf_weight_norm": 0.0,
                    "selection_pass": None,
                }]
                stored_results[h] = {
                    "result": {
                        "original_input": str(sample.get("input", "") or ""),
                        "original_output": str(sample.get("output", "") or ""),
                        "term_candidates_influential": [],
                        "tfidf_weights_raw": {},
                        "tfidf_weights_norm": {},
                        "counterfactuals_robust": err_cfs,
                        "counterfactuals_strict": [],
                        "per_type_status": {"syn": "PROCESS_SAMPLE_EXCEPTION"},
                    },
                    "meta": {"prompt_sig": PROMPT_SIG},
                }
                if (idx1) % CACHE_FLUSH_EVERY == 0:
                    try:
                        with open(CACHE_FILE, "wb") as f:
                            pickle.dump(stored_results, f)
                    except Exception:
                        pass
                continue

            stored_results[h] = {"result": res, "meta": {"prompt_sig": PROMPT_SIG}}

        if PRINT_EVERY and (idx1 % PRINT_EVERY == 0):
            _print_sample(res, idx1, total, h)

        orig_out = str(res.get("original_output", "") or "")
        orig_in = str(res.get("original_input", "") or "")

        orig_out_emb = _bert_embed(orig_out) if EVAL_BERT else None
        orig_in_emb = _bert_embed(orig_in) if EVAL_BERT else None

        strict_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for scf in res.get("counterfactuals_strict", []) or []:
            k = (str(scf.get("type")), str(scf.get("term")), str(scf.get("replacement")))
            strict_index[k] = scf

        cache_dirty = False

        for cf in res.get("counterfactuals_robust", []) or []:
            typ = (cf.get("type") or "").strip()
            if typ not in types:
                continue
            counters["attempts"][typ] += 1
            st = str(cf.get("status") or "UNKNOWN")
            counters["status_counts"][typ][st] += 1
            counters["strict_valid"][typ]["true" if bool(cf.get("strict_valid")) else "false"] += 1
            counters["ungrammatical"][typ]["true" if bool(cf.get("ungrammatical_proxy")) else "false"] += 1

        for cf in res.get("counterfactuals_robust", []) or []:
            key_out = "syn_out_vs_orig_out"
            key_in = "syn_in_vs_orig_in"

            cf_out = str((cf.get("final_advice") or "").strip())
            is_strict = bool(cf.get("strict_valid"))

            if cf_out:
                if EVAL_BERT and orig_out_emb is not None:
                    cf_out_emb = _bert_embed(cf_out)
                    b = cosine_bert_from_emb(cf_out_emb, orig_out_emb)
                    imp = effect_size_from_emb(cf_out_emb, orig_out_emb)
                else:
                    b, imp = 0.0, None

                if imp is not None and cf.get("impact_1_minus_bert") != float(imp):
                    cf["impact_1_minus_bert"] = float(imp)
                    cache_dirty = True

                outR[key_out]["cosine"].append(tfidf_cosine(cf_out, orig_out))
                outR[key_out]["seq"].append(SequenceMatcher(None, cf_out, orig_out).ratio())
                outR[key_out]["bert"].append(b)
                outR[key_out]["rougeL"].append(rouge_l(cf_out, orig_out))
                outR[key_out]["meteor"].append(meteor(cf_out, orig_out))
                outR[key_out]["nli"].append(cf.get("nli_proxy", "UNKNOWN"))
                if imp is not None:
                    outR[key_out]["impact"].append(float(imp))

                if is_strict:
                    outS[key_out]["cosine"].append(outR[key_out]["cosine"][-1])
                    outS[key_out]["seq"].append(outR[key_out]["seq"][-1])
                    outS[key_out]["bert"].append(outR[key_out]["bert"][-1])
                    outS[key_out]["rougeL"].append(outR[key_out]["rougeL"][-1])
                    outS[key_out]["meteor"].append(outR[key_out]["meteor"][-1])
                    outS[key_out]["nli"].append(outR[key_out]["nli"][-1])
                    if imp is not None:
                        outS[key_out]["impact"].append(float(imp))

                    k = (str(cf.get("type")), str(cf.get("term")), str(cf.get("replacement")))
                    scf = strict_index.get(k)
                    if scf is not None and imp is not None and scf.get("impact_1_minus_bert") != float(imp):
                        scf["impact_1_minus_bert"] = float(imp)
                        cache_dirty = True

            mod_in = str((cf.get("modified_input") or "").strip())
            if mod_in:
                if EVAL_BERT and orig_in_emb is not None:
                    mod_in_emb = _bert_embed(mod_in)
                    b_in = cosine_bert_from_emb(mod_in_emb, orig_in_emb)
                else:
                    b_in = 0.0

                inR[key_in]["cosine"].append(tfidf_cosine(mod_in, orig_in))
                inR[key_in]["seq"].append(SequenceMatcher(None, mod_in, orig_in).ratio())
                inR[key_in]["bert"].append(b_in)
                inR[key_in]["rougeL"].append(rouge_l(mod_in, orig_in))
                inR[key_in]["meteor"].append(meteor(mod_in, orig_in))

                if is_strict:
                    inS[key_in]["cosine"].append(inR[key_in]["cosine"][-1])
                    inS[key_in]["seq"].append(inR[key_in]["seq"][-1])
                    inS[key_in]["bert"].append(inR[key_in]["bert"][-1])
                    inS[key_in]["rougeL"].append(inR[key_in]["rougeL"][-1])
                    inS[key_in]["meteor"].append(inR[key_in]["meteor"][-1])

        if cache_dirty:
            stored_results[h] = {"result": res, "meta": {"prompt_sig": PROMPT_SIG}}

        if (idx1) % CACHE_FLUSH_EVERY == 0:
            try:
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(stored_results, f)
            except Exception:
                pass

    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(stored_results, f)
    except Exception:
        pass

    def mean(x): return float(np.mean(x)) if x else None
    def majority(x): return max(set(x), key=x.count) if x else None

    def pack_out(b):
        return {
            "cosine": mean(b["cosine"]),
            "seq": mean(b["seq"]),
            "bert": mean(b["bert"]),
            "rougeL": mean(b["rougeL"]),
            "meteor": mean(b["meteor"]),
            "nli_proxy_majority": majority(b["nli"]),
            "impact_mean": mean(b["impact"]),
        }

    def pack_in(b):
        return {
            "cosine": mean(b["cosine"]),
            "seq": mean(b["seq"]),
            "bert": mean(b["bert"]),
            "rougeL": mean(b["rougeL"]),
            "meteor": mean(b["meteor"]),
        }

    error_analysis = _finalize_error_analysis(counters, types)

    return {
        "robust_out": {k: pack_out(v) for k, v in outR.items()},
        "strict_out": {k: pack_out(v) for k, v in outS.items()},
        "robust_in": {k: pack_in(v) for k, v in inR.items()},
        "strict_in": {k: pack_in(v) for k, v in inS.items()},
        "meta": {
            "mode": mode,
            "device": str(DEVICE),
            "seed": int(SEED),
            "cache_file": CACHE_FILE,
            "prompt_sig": PROMPT_SIG,
            "max_items": int(max_items),
            "eval_bert": bool(EVAL_BERT),
            "eval_nli": bool(EVAL_NLI),
            "eval_rouge": bool(EVAL_ROUGE),
            "eval_meteor": bool(EVAL_METEOR),
            "versions": get_versions(),
            "error_analysis": error_analysis,
        },
    }


#main
if __name__ == "__main__":
    nltk_preflight()

    with open("cleaned_healthcaremagic.pkl", "rb") as f:
        cleaned_healthcaremagic = pickle.load(f)

    print(f"[DATA] Loaded dataset with {len(cleaned_healthcaremagic)} samples")
    print(f"[META] seed={SEED} device={DEVICE} versions={get_versions()}")
    print(f"[RUN_TAG] {RUN_TAG}")
    print(f"[PROMPT_SIG] {PROMPT_SIG}")
    print(f"[CACHE_MODE] FORCE_NO_CACHE={FORCE_NO_CACHE} STRICT_CACHE_PROMPT_SIG={STRICT_CACHE_PROMPT_SIG}")

    rb = rss_bytes()
    if rb is not None:
        print(f"[BOOT] RSS={rb/1024/1024:.1f}MB")

    if ERROR_ANALYSIS_ONLY:
        ea = error_analysis_from_cache(stored_results, mode=MODE_TO_RUN)
        print("\n\nERROR ANALYSIS (FROM CACHE ONLY)")
        print(json.dumps(ea, indent=2))
        raise SystemExit(0)

    _TFIDF_VECTORIZER = build_tfidf_vectorizer([str(s.get("input", "") or "") for s in cleaned_healthcaremagic])
    print(f"[TFIDF] fit corpus size: {len(cleaned_healthcaremagic)}")

    if EVAL_BERT:
        _get_bert_model()
    if EVAL_NLI:
        _get_nli()

    try:
        if EVAL_BERT:
            _ = _bert_embed("warmup")
        if EVAL_NLI:
            _ = nli_proxy_label("warmup premise", "warmup hypothesis")
        try:
            _ = timed_llm_invoke("warmup[influential]", influential_chain, {"input": "warmup question"})
        except Exception:
            pass
    except Exception:
        pass

    results = evaluate_dataset(cleaned_healthcaremagic, max_items=MAX_ITEMS, mode=MODE_TO_RUN)

    print("\n\n" + "=" * 100)
    print("FINAL AGGREGATED RESULTS (ALL BUCKETS/METRICS)")
    print(json.dumps(results, indent=2))

    print("\n\n" + "=" * 100)
    print("ERROR ANALYSIS SUMMARY (ONLINE)")
    print(json.dumps(results["meta"]["error_analysis"], indent=2))


#ERROR ANALYSIS
#Separates:
#(1) Coverage (% CF_GENERATED)
#(2) Pipeline failure (% BAD_STATUSES)
#(3) Strict validity (% strict_valid among covered)
#(4) Balanced semantic fidelity on STRICT-VALID only:
#SYN: risk buckets (LOW/MED/HIGH)&hard_error (CONTRADICTION or very high impact)
#ANT: hard_error (ENTAILMENT or too-low impact)

#Adds WEIGHTS:
#mean/median weights per group
#weighted error rate = sum(weights of errors)/sum(weights total)
#top influential terms driving errors (weighted)
# ant_subtype breakdown (antonym vs negation)


#CONFIG
PHASES = ["SYN_ONLY", "ANT_ONLY", "BOTH"]

CACHE_FILES = {
    "SYN_ONLY": "stored_results_expl_SYN_ONLY_synonly_FINAL_scientific_coverage_safe_bert1_nli1_met1_rouge1.pkl",
    "ANT_ONLY": "stored_results_expl_ANT_ONLY_bert1_nli1_met1_rouge1.pkl",
    "BOTH":     "stored_results_expl_BOTH_bert1_nli1_met1_rouge1.pkl",
}

FIG_DIR = "Figures"
OUT_DIR = "Section9_outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

BAD_STATUSES = {
    "NO_INFLUENTIAL_TERMS",
    "NO_MATCH",
    "NO_FINAL_ADVICE",
    "PROCESS_SAMPLE_EXCEPTION",
}

DEFAULT_TAU_SYN = 0.25  #fallback for "very high change" in syn
DEFAULT_TAU_ANT = 0.10  #fallback for "too low change" in ant


#helpers functions
def to_float_or_nan(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def norm_nli(x: str) -> str:
    x = str(x or "UNKNOWN").strip().upper()
    if "ENTAIL" in x:
        return "ENTAILMENT"
    if "CONTRAD" in x:
        return "CONTRADICTION"
    if "NEUT" in x:
        return "NEUTRAL"
    return x if x else "UNKNOWN"

def syn_risk_bucket(nli: str) -> str:
    # Balanced: NEUTRAL is uncertainty, not error
    if nli == "ENTAILMENT":     return "LOW"
    if nli == "NEUTRAL":        return "MEDIUM"
    if nli == "CONTRADICTION":  return "HIGH"
    return "UNKNOWN"

def safe_weight(x) -> float:
    try:
        v = float(x)
        if np.isnan(v) or v < 0:
            return 0.0
        return v
    except Exception:
        return 0.0

def weighted_rate(df: pd.DataFrame, flag_col: str, weight_col: str = "weight") -> float:
    # weighted rate = sum(w for flag==True)/sum(w total)
    if len(df) == 0:
        return np.nan
    w = df[weight_col].astype(float).fillna(0.0).clip(lower=0.0).values
    y = df[flag_col].astype(bool).values
    denom = float(np.sum(w))
    if denom <= 0:
        return np.nan
    return float(np.sum(w[y]) / denom)

def plot_hist(series, title, xlabel, filename, bins=20):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(s.values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print("[INFO] Saved:", out)

def plot_two_bars(pv, title, ylabel, filename):
    x = np.arange(len(pv.index))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9,4))

    syn_vals = pv.get("syn", pd.Series([0]*len(x), index=pv.index)).values
    ant_vals = pv.get("ant", pd.Series([0]*len(x), index=pv.index)).values

    ax.bar(x - w/2, syn_vals, w, label="syn")
    ax.bar(x + w/2, ant_vals, w, label="ant")

    ax.set_xticks(x)
    ax.set_xticklabels(pv.index)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    for i,v in enumerate(syn_vals):
        if v > 0:
            ax.text(i - w/2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for i,v in enumerate(ant_vals):
        if v > 0:
            ax.text(i + w/2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print("[INFO] Saved:", out)

def pivot_metric(df, value_col, index_col="phase", col_col="type"):
    pv = df.pivot_table(index=index_col, columns=col_col, values=value_col, aggfunc="mean").fillna(0.0)
    pv = pv.loc[[p for p in PHASES if p in pv.index]]
    return pv


#load caches
stored_by_phase = {}
for phase in PHASES:
    fn = CACHE_FILES[phase]
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Missing cache file for phase={phase}: {fn}")
    with open(fn, "rb") as f:
        stored_by_phase[phase] = pickle.load(f) or {}
    print(f"[INFO] Loaded {fn}: {len(stored_by_phase[phase])} cached samples")


#build df_cf(cache-driven)
rows = []
for phase in PHASES:
    for sample_id, entry in stored_by_phase[phase].items():
        res = (entry or {}).get("result", {}) or {}

        orig_in  = str(res.get("original_input", "") or "")
        orig_out = str(res.get("original_output", "") or "")

        for cf in (res.get("counterfactuals_robust", []) or []):
            cf_type = (cf.get("type") or "").strip()
            if cf_type not in {"syn", "ant"}:
                continue

            rows.append({
                "phase": phase,
                "sample_id": str(sample_id),
                "type": cf_type,

                "term": cf.get("term", ""),
                "replacement": cf.get("replacement", ""),

                "orig_input": orig_in,
                "orig_output": orig_out,

                "modified_input": cf.get("modified_input", ""),
                "final_advice": cf.get("final_advice", ""),

                "status": str(cf.get("status", "") or "").strip(),
                "strict_valid": bool(cf.get("strict_valid", False)),
                "ungrammatical_proxy": bool(cf.get("ungrammatical_proxy", False)),
                "ant_subtype": cf.get("ant_subtype", None),

                "weight": safe_weight(cf.get("tfidf_weight_norm", 0.0) or 0.0),
                "weight_raw": safe_weight(cf.get("tfidf_weight_raw", 0.0) or 0.0),

                "nli_proxy": (cf.get("nli_proxy", None) or "UNKNOWN"),
                "impact": to_float_or_nan(cf.get("impact_1_minus_bert", None)),  # 1 - bertcos
            })

df_cf = pd.DataFrame(rows)
print(f"[INFO] df_cf rows: {len(df_cf)}")
print(df_cf[["phase","type"]].value_counts())

df_cf.to_csv(os.path.join(OUT_DIR, "df_counterfactuals_RAW.csv"), index=False)


#pipeline levels
df_cf["pipeline_fail"] = df_cf["status"].isin(BAD_STATUSES)
df_cf["covered"] = df_cf["status"].eq("CF_GENERATED")
df_cf["has_advice"] = df_cf["final_advice"].astype(str).str.strip().str.len() > 0

df_cf["usable_strict"] = (~df_cf["pipeline_fail"]) & (df_cf["covered"]) & (df_cf["strict_valid"]) & (df_cf["has_advice"])




#NLI normalization&risk buckets
df_cf["nli_norm"] = df_cf["nli_proxy"].apply(norm_nli)
df_cf["risk_bucket"] = ""
mask_syn = df_cf["type"].eq("syn")
df_cf.loc[mask_syn, "risk_bucket"] = df_cf.loc[mask_syn, "nli_norm"].apply(syn_risk_bucket)


#thresholds from STRICT-VALID only
df_u = df_cf[df_cf["usable_strict"]].copy()

syn_imp = df_u[(df_u["type"] == "syn") & (df_u["impact"].notna())]["impact"].values
ant_imp = df_u[(df_u["type"] == "ant") & (df_u["impact"].notna())]["impact"].values

TAU_SYN = float(np.quantile(syn_imp, 0.75)) if len(syn_imp) >= 10 else DEFAULT_TAU_SYN
TAU_ANT = float(np.quantile(ant_imp, 0.25)) if len(ant_imp) >= 10 else DEFAULT_TAU_ANT

print(f"[THRESHOLDS] TAU_SYN (very high change) = {TAU_SYN:.4f}")
print(f"[THRESHOLDS] TAU_ANT (too low change)  = {TAU_ANT:.4f}")




#balanced HARD semantic error (STRICT ONLY)
def is_hard_semantic_error(row) -> bool:
    if not bool(row["usable_strict"]):
        return False
    typ = row["type"]
    nli = row["nli_norm"]
    imp = row["impact"]
    has_imp = not (isinstance(imp, float) and np.isnan(imp))

    if typ == "syn":
        return (nli == "CONTRADICTION") or (has_imp and float(imp) > TAU_SYN)

    if typ == "ant":
        return (nli == "ENTAILMENT") or (has_imp and float(imp) < TAU_ANT)

    return False

df_cf["hard_semantic_error"] = df_cf.apply(is_hard_semantic_error, axis=1)

def hard_error_reason(row) -> str:
    if not bool(row["usable_strict"]):
        return ""
    typ = row["type"]
    nli = row["nli_norm"]
    imp = row["impact"]
    has_imp = not (isinstance(imp, float) and np.isnan(imp))

    reasons = []
    if typ == "syn":
        if nli == "CONTRADICTION":
            reasons.append("NLI=CONTRADICTION")
        if has_imp and float(imp) > TAU_SYN:
            reasons.append(f"impact>{TAU_SYN:.3f}")
    elif typ == "ant":
        if nli == "ENTAILMENT":
            reasons.append("NLI=ENTAILMENT")
        if has_imp and float(imp) < TAU_ANT:
            reasons.append(f"impact<{TAU_ANT:.3f}")
    return "|".join(reasons)

df_cf["hard_error_reason"] = df_cf.apply(hard_error_reason, axis=1)

df_cf.to_csv(os.path.join(OUT_DIR, "df_counterfactuals_balanced_rich.csv"), index=False)
print(f"[INFO] Saved: {OUT_DIR}/df_counterfactuals_balanced_rich.csv")


#Pipeline
coverage_tbl = (
    df_cf.groupby(["phase","type"])
        .agg(total=("covered","size"), covered=("covered","sum"))
        .assign(coverage_rate_pct=lambda x: 100 * x["covered"]/x["total"])
        .reset_index()
)
pipeline_tbl = (
    df_cf.groupby(["phase","type"])
        .agg(total=("pipeline_fail","size"), fails=("pipeline_fail","sum"))
        .assign(pipeline_failure_rate_pct=lambda x: 100 * x["fails"]/x["total"])
        .reset_index()
)
strict_tbl_cov = (
    df_cf[df_cf["covered"]]
        .groupby(["phase","type"])
        .agg(total=("strict_valid","size"), strict=("strict_valid","sum"))
        .assign(strict_valid_rate_pct=lambda x: 100 * x["strict"]/x["total"])
        .reset_index()
)

status_counts = (
    df_cf.groupby(["phase","type","status"])
         .size()
         .reset_index(name="count")
)
status_rates = status_counts.merge(
    df_cf.groupby(["phase","type"]).size().reset_index(name="total"),
    on=["phase","type"],
    how="left"
)
status_rates["rate_pct"] = 100 * status_rates["count"] / status_rates["total"]

coverage_tbl.to_csv(os.path.join(OUT_DIR, "coverage_rates.csv"), index=False)
pipeline_tbl.to_csv(os.path.join(OUT_DIR, "pipeline_failure_rates.csv"), index=False)
strict_tbl_cov.to_csv(os.path.join(OUT_DIR, "strict_valid_rates_ON_COVERED.csv"), index=False)
status_counts.to_csv(os.path.join(OUT_DIR, "status_counts_by_phase_type.csv"), index=False)
status_rates.to_csv(os.path.join(OUT_DIR, "status_rates_by_phase_type.csv"), index=False)

print("\n=== Coverage (%) ===\n", coverage_tbl)
print("\n=== Pipeline failures (%) ===\n", pipeline_tbl)
print("\n=== Strict-valid (%) among covered ===\n", strict_tbl_cov)



#TABLES-(Model behavior) STRICT-ONLY
df_sem = df_cf[df_cf["usable_strict"]].copy()

hard_sem_tbl = (
    df_sem.groupby(["phase","type"])
        .agg(total=("hard_semantic_error","size"),
             hard_errors=("hard_semantic_error","sum"),
             weighted_error_rate=("hard_semantic_error", lambda s: np.nan))  # placeholder
        .reset_index()
)
# fill weighted rates
hard_sem_tbl["weighted_error_rate"] = hard_sem_tbl.apply(
    lambda r: weighted_rate(
        df_sem[(df_sem["phase"]==r["phase"]) & (df_sem["type"]==r["type"])],
        "hard_semantic_error",
        "weight"
    ),
    axis=1
)
hard_sem_tbl["hard_error_rate_per_100"] = 100 * hard_sem_tbl["hard_errors"] / hard_sem_tbl["total"]
hard_sem_tbl["weighted_error_rate_per_100"] = 100 * hard_sem_tbl["weighted_error_rate"]

hard_sem_tbl.to_csv(os.path.join(OUT_DIR, "hard_semantic_error_rates_STRICT_ONLY.csv"), index=False)

#SYN risk buckets (strict-only)
risk_tbl = (
    df_sem[df_sem["type"].eq("syn")]
        .groupby(["phase","risk_bucket"])
        .size()
        .reset_index(name="count")
)
risk_tbl.to_csv(os.path.join(OUT_DIR, "syn_risk_buckets_STRICT_ONLY.csv"), index=False)



#NLI distribution (strict-only)
nli_tbl = (
    df_sem.groupby(["phase","type","nli_norm"])
        .size()
        .reset_index(name="count")
)
nli_tbl.to_csv(os.path.join(OUT_DIR, "nli_distribution_STRICT_ONLY.csv"), index=False)



#weight stats (strict-only)
weight_stats = (
    df_sem.groupby(["phase","type"])
          .agg(
              n=("weight","size"),
              weight_mean=("weight","mean"),
              weight_median=("weight","median"),
              weight_std=("weight","std"),
              weight_p90=("weight", lambda s: float(np.quantile(s, 0.90)) if len(s) else np.nan),
          )
          .reset_index()
)
weight_stats.to_csv(os.path.join(OUT_DIR, "weight_stats_STRICT_ONLY.csv"), index=False)

# Weight vs hard error (strict-only)
weight_by_err = (
    df_sem.groupby(["phase","type","hard_semantic_error"])["weight"]
         .agg(["size","mean","median","std"])
         .reset_index()
         .rename(columns={"size":"n"})
)
weight_by_err.to_csv(os.path.join(OUT_DIR, "weight_by_hard_error_STRICT_ONLY.csv"), index=False)





#ANT subtype breakdown (strict-only)
df_ant_sem = df_sem[df_sem["type"].eq("ant")].copy()
if len(df_ant_sem) > 0:
    ant_sub_tbl = (
        df_ant_sem.groupby(["phase","ant_subtype"])
                  .agg(
                      n=("hard_semantic_error","size"),
                      hard_errors=("hard_semantic_error","sum"),
                      hard_error_rate_per_100=("hard_semantic_error", lambda s: 100*float(np.mean(s)) if len(s) else np.nan),
                      weight_mean=("weight","mean"),
                      weighted_error_rate=("hard_semantic_error", lambda s: np.nan)
                  )
                  .reset_index()
    )
    ant_sub_tbl["weighted_error_rate"] = ant_sub_tbl.apply(
        lambda r: weighted_rate(
            df_ant_sem[(df_ant_sem["phase"]==r["phase"]) & (df_ant_sem["ant_subtype"]==r["ant_subtype"])],
            "hard_semantic_error",
            "weight"
        ),
        axis=1
    )
    ant_sub_tbl["weighted_error_rate_per_100"] = 100*ant_sub_tbl["weighted_error_rate"]
    ant_sub_tbl.to_csv(os.path.join(OUT_DIR, "ant_subtype_breakdown_STRICT_ONLY.csv"), index=False)
else:
    ant_sub_tbl = pd.DataFrame(columns=["phase","ant_subtype","n","hard_errors"])



#top terms driving hard errors (weighted) STRICT-ONLY
df_err = df_sem[df_sem["hard_semantic_error"]].copy()
top_terms = (
    df_err.groupby(["phase","type","term"])
          .agg(
              n=("term","size"),
              weight_sum=("weight","sum"),
              weight_mean=("weight","mean"),
              impact_mean=("impact","mean")
          )
          .reset_index()
          .sort_values(["phase","type","weight_sum"], ascending=[True, True, False])
)
top_terms.to_csv(os.path.join(OUT_DIR, "top_terms_driving_hard_errors_STRICT_ONLY.csv"), index=False)

print("\n=== HARD semantic error rate per 100 (STRICT only) ===\n", hard_sem_tbl)
print("\n=== Weight stats (STRICT only) ===\n", weight_stats)
if len(ant_sub_tbl) > 0:
    print("\n=== ANT subtype breakdown (STRICT only) ===\n", ant_sub_tbl)



#FIGURES-Pipeline
plot_two_bars(pivot_metric(coverage_tbl, "coverage_rate_pct"),
              "Coverage Rate by Phase and Type (status == CF_GENERATED)", "Coverage (%)",
              "fig_cov_rate_by_phase_type.png")

plot_two_bars(pivot_metric(pipeline_tbl, "pipeline_failure_rate_pct"),
              "Pipeline Failure Rate by Phase and Type", "Pipeline failures (%)",
              "fig_pipeline_failure_rate_by_phase_type.png")

plot_two_bars(pivot_metric(strict_tbl_cov, "strict_valid_rate_pct"),
              "Strict Validity Rate by Phase and Type (among covered only)", "Strict-valid (%)",
              "fig_strict_valid_on_covered_by_phase_type.png")




#FIGURES-balanced semantic(STRICT only)
plot_two_bars(
    pivot_metric(hard_sem_tbl, "hard_error_rate_per_100", index_col="phase", col_col="type"),
    "Hard Semantic Error Rate per 100 (Strict-valid Only, Balanced)", "Hard errors per 100",
    "fig_hard_semantic_error_rate_STRICT_ONLY.png"
)

plot_two_bars(
    pivot_metric(hard_sem_tbl, "weighted_error_rate_per_100", index_col="phase", col_col="type"),
    "WEIGHTED Hard Semantic Error Rate per 100 (Strict-valid Only)", "Weighted hard errors per 100",
    "fig_weighted_hard_semantic_error_rate_STRICT_ONLY.png"
)

#NLI distribution plot per phase (strict-only)
if len(df_sem) > 0:
    ct_nli = pd.crosstab([df_sem["phase"], df_sem["type"]], df_sem["nli_norm"])
    cols = [c for c in ["ENTAILMENT","NEUTRAL","CONTRADICTION","UNKNOWN"] if c in ct_nli.columns]
    ct_nli = ct_nli[cols]
    ct_nli.to_csv(os.path.join(OUT_DIR, "nli_crosstab_STRICT_ONLY.csv"))

    for ph in PHASES:
        if ph not in ct_nli.index.get_level_values(0):
            continue
        sub = ct_nli.loc[ph]
        labels = cols
        x = np.arange(len(labels))
        w = 0.35

        syn_vals = sub.loc["syn"].values if "syn" in sub.index else np.zeros(len(labels))
        ant_vals = sub.loc["ant"].values if "ant" in sub.index else np.zeros(len(labels))

        fig, ax = plt.subplots(figsize=(9,4))
        b1 = ax.bar(x - w/2, syn_vals, w, label="Synonym")
        b2 = ax.bar(x + w/2, ant_vals, w, label="Antonym")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count (strict-valid only)")
        ax.set_title(f"NLI Proxy Distribution (Strict-valid Only) — {ph}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend()

        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{int(h)}",
                        ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"fig_nli_distribution_STRICT_ONLY_{ph}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()
        print("[INFO] Saved:", out)



#SYN risk buckets plot per phase (strict-only)
df_syn_sem = df_sem[df_sem["type"].eq("syn")].copy()
if len(df_syn_sem) > 0:
    ct_risk = pd.crosstab(df_syn_sem["phase"], df_syn_sem["risk_bucket"])
    risk_cols = [c for c in ["LOW","MEDIUM","HIGH","UNKNOWN"] if c in ct_risk.columns]
    ct_risk = ct_risk[risk_cols]
    ct_risk.to_csv(os.path.join(OUT_DIR, "syn_risk_crosstab_STRICT_ONLY.csv"))

    for ph in PHASES:
        if ph not in ct_risk.index:
            continue
        vals = ct_risk.loc[ph].values
        labels = ct_risk.columns.tolist()
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count (syn strict-valid only)")
        ax.set_title(f"SYN Risk Buckets (Strict-valid Only) — {ph}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for i,v in enumerate(vals):
            ax.text(i, v + 0.5, f"{int(v)}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"fig_syn_risk_buckets_STRICT_ONLY_{ph}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()
        print("[INFO] Saved:", out)





#FIGURES-weights&impact (STRICT only)
#impact histograms per type
plot_hist(df_sem[df_sem["type"].eq("syn")]["impact"],
          "Impact Distribution (SYN, strict-valid only)", "impact = 1 - BERTcos",
          "fig_impact_hist_syn_STRICT_ONLY.png", bins=18)

plot_hist(df_sem[df_sem["type"].eq("ant")]["impact"],
          "Impact Distribution (ANT, strict-valid only)", "impact = 1 - BERTcos",
          "fig_impact_hist_ant_STRICT_ONLY.png", bins=18)

#weight histograms by hard error
plot_hist(df_sem[~df_sem["hard_semantic_error"]]["weight"],
          "Weight Distribution (No Hard Error, strict-valid only)", "tfidf_weight_norm",
          "fig_weight_hist_no_hard_error_STRICT_ONLY.png", bins=18)

plot_hist(df_sem[df_sem["hard_semantic_error"]]["weight"],
          "Weight Distribution (Hard Error, strict-valid only)", "tfidf_weight_norm",
          "fig_weight_hist_hard_error_STRICT_ONLY.png", bins=18)

#scatter: weight vs impact (strict-only) with hard_error overlay count
if len(df_sem) > 0 and df_sem["impact"].notna().any():
    df_sc = df_sem[df_sem["impact"].notna()].copy()
    fig, ax = plt.subplots(figsize=(7,5))

    a = df_sc[~df_sc["hard_semantic_error"]]
    b = df_sc[df_sc["hard_semantic_error"]]

    ax.scatter(a["weight"].values, a["impact"].values, label="No hard error", alpha=0.6)
    ax.scatter(b["weight"].values, b["impact"].values, label="Hard error", alpha=0.8)

    ax.set_xlabel("tfidf_weight_norm")
    ax.set_ylabel("impact = 1 - BERTcos")
    ax.set_title("Weight vs Impact (Strict-valid only)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_scatter_weight_vs_impact_STRICT_ONLY.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print("[INFO] Saved:", out)





#input similarity stats (TFIDF/SEQ) per phase
def tfidf_cos(a: str, b: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer().fit([a, b])
        vecs = vect.transform([a, b])
        return float((vecs[0] @ vecs[1].T).toarray()[0][0])
    except Exception:
        return np.nan

def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a), str(b)).ratio()

df_cf["input_tfidf"] = np.nan
df_cf["input_seq"] = np.nan

mask_mod = df_cf["modified_input"].astype(str).str.len() > 0
df_cf.loc[mask_mod, "input_tfidf"] = df_cf.loc[mask_mod].apply(
    lambda r: tfidf_cos(r["orig_input"], r["modified_input"]), axis=1
)
df_cf.loc[mask_mod, "input_seq"] = df_cf.loc[mask_mod].apply(
    lambda r: seq_sim(r["orig_input"], r["modified_input"]), axis=1
)

lex_stats = (
    df_cf.groupby(["phase","type"])[["input_tfidf","input_seq"]]
         .agg(["mean","std"])
)
lex_stats.to_csv(os.path.join(OUT_DIR, "input_similarity_stats_by_phase_and_type.csv"))
print("\n=== Input similarity stats (TFIDF/SEQ) by phase+type ===\n", lex_stats)




#final summary PKL
final_metrics = {
    "thresholds": {"tau_syn": float(TAU_SYN), "tau_ant": float(TAU_ANT)},
    "coverage_rates": coverage_tbl.to_dict(orient="records"),
    "pipeline_failure_rates": pipeline_tbl.to_dict(orient="records"),
    "strict_valid_rates_on_covered": strict_tbl_cov.to_dict(orient="records"),
    "status_counts_by_phase_type": status_counts.to_dict(orient="records"),
    "status_rates_by_phase_type": status_rates.to_dict(orient="records"),

    "hard_semantic_error_rates_strict_only": hard_sem_tbl.to_dict(orient="records"),
    "syn_risk_bucket_distribution_strict_only": risk_tbl.to_dict(orient="records"),
    "nli_distribution_strict_only": nli_tbl.to_dict(orient="records"),

    "weight_stats_strict_only": weight_stats.to_dict(orient="records"),
    "weight_by_hard_error_strict_only": weight_by_err.to_dict(orient="records"),
    "ant_subtype_breakdown_strict_only": ant_sub_tbl.to_dict(orient="records") if len(ant_sub_tbl) else [],

    "top_terms_driving_hard_errors_strict_only": top_terms.head(200).to_dict(orient="records"),
}

with open(os.path.join(OUT_DIR, "final_metrics_summary_balanced_rich.pkl"), "wb") as f:
    pickle.dump(final_metrics, f)

print(f"\n[INFO] Saved: {OUT_DIR}/final_metrics_summary_balanced_rich.pkl")
print(f"[INFO] Figures saved in: {FIG_DIR}/")
print(f"[INFO] Tables/CSVs saved in: {OUT_DIR}/")


#ERROR ANALYSIS (phase-aware) 
#Coverage / Pipeline failures / Strict-valid among covered
#Error-type proxies computed ONLY on usable_strict
#(so pipeline failures don't inflate "model behavior" errors)



# CONFIG
PHASES = ["SYN_ONLY", "ANT_ONLY", "BOTH"]

CACHE_FILES = {
    "SYN_ONLY": "stored_results_expl_SYN_ONLY_synonly_FINAL_scientific_coverage_safe_bert1_nli1_met1_rouge1.pkl",
    "ANT_ONLY": "stored_results_expl_ANT_ONLY_bert1_nli1_met1_rouge1.pkl",
    "BOTH":     "stored_results_expl_BOTH_bert1_nli1_met1_rouge1.pkl",
}

FIG_DIR = "Figures"
OUT_DIR = "Section9_outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


#helpers functions
def normalize_text(x: str) -> str:
    x = str(x or "").lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x

def to_float_or_nan(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan



#load caches
stored_by_phase = {}
for phase in PHASES:
    pkl = CACHE_FILES[phase]
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Missing cache for {phase}: {pkl}")
    with open(pkl, "rb") as f:
        stored_by_phase[phase] = pickle.load(f) or {}
    print(f"[INFO] Loaded {phase}: {pkl} ({len(stored_by_phase[phase])} samples)")



#build df_cf from cache (counterfactuals_robust)
rows = []

for phase in PHASES:
    stored_results = stored_by_phase[phase]

    for sample_id, entry in stored_results.items():
        res = (entry or {}).get("result", {}) or {}

        orig_in  = str(res.get("original_input", "") or "")
        orig_out = str(res.get("original_output", "") or "")

        cfr = res.get("counterfactuals_robust", []) or []
        for cf in cfr:
            cf_type = (cf.get("type") or "").strip()  # syn / ant
            if cf_type not in {"syn", "ant"}:
                continue

            rows.append({
                "phase": phase,
                "sample_id": str(sample_id),
                "type": cf_type,

                "term": cf.get("term", ""),
                "replacement": cf.get("replacement", ""),

                "orig_input": orig_in,
                "orig_output": orig_out,

                "modified_input": cf.get("modified_input", ""),
                "final_advice": cf.get("final_advice", ""),

                "status": str(cf.get("status", "") or "").strip(),
                "strict_valid": bool(cf.get("strict_valid", False)),
                "ungrammatical_proxy": bool(cf.get("ungrammatical_proxy", False)),
                "ant_subtype": cf.get("ant_subtype", None),

                "weight": float(cf.get("tfidf_weight_norm", 0.0) or 0.0),
                "weight_raw": float(cf.get("tfidf_weight_raw", 0.0) or 0.0),

                # if you have these in cache, keep them (optional)
                "nli_proxy": (cf.get("nli_proxy", None) or "UNKNOWN"),
                "impact": to_float_or_nan(cf.get("impact_1_minus_bert", None)),
            })

df_cf = pd.DataFrame(rows)
print(f"\n[INFO] df_cf rows: {len(df_cf)}")
print(df_cf[["phase", "type"]].value_counts())

df_cf.to_csv(os.path.join(OUT_DIR, "df_counterfactuals_all_phases_RAW.csv"), index=False)
print(f"[INFO] Saved RAW: {OUT_DIR}/df_counterfactuals_all_phases_RAW.csv")



#pipeline levels (Coverage / Failures / Strict-valid)
BAD_STATUSES = {
    "NO_INFLUENTIAL_TERMS",
    "NO_MATCH",
    "NO_FINAL_ADVICE",
    "PROCESS_SAMPLE_EXCEPTION",
}

df_cf["pipeline_fail"] = df_cf["status"].isin(BAD_STATUSES)
df_cf["has_advice"] = df_cf["final_advice"].astype(str).str.strip().str.len() > 0
df_cf["covered"] = df_cf["status"].eq("CF_GENERATED")

#usable_strict = strict-valid + covered + advice present + not pipeline_fail
df_cf["usable_strict"] = (~df_cf["pipeline_fail"]) & (df_cf["covered"]) & (df_cf["strict_valid"]) & (df_cf["has_advice"])

#tables: coverage / pipeline failure / strict-valid among covered
coverage_tbl = (
    df_cf.groupby(["phase", "type"])
         .agg(total=("covered", "size"), covered=("covered", "sum"))
         .assign(coverage_rate_pct=lambda x: (x["covered"]/x["total"])*100)
         .reset_index()
)
pipeline_tbl = (
    df_cf.groupby(["phase", "type"])
         .agg(total=("pipeline_fail", "size"), fails=("pipeline_fail", "sum"))
         .assign(pipeline_failure_rate_pct=lambda x: (x["fails"]/x["total"])*100)
         .reset_index()
)
strict_cov_tbl = (
    df_cf[df_cf["covered"]]
    .groupby(["phase", "type"])
    .agg(total=("strict_valid", "size"), strict=("strict_valid", "sum"))
    .assign(strict_valid_rate_pct=lambda x: (x["strict"]/x["total"])*100)
    .reset_index()
)

coverage_tbl.to_csv(os.path.join(OUT_DIR, "coverage_rates.csv"), index=False)
pipeline_tbl.to_csv(os.path.join(OUT_DIR, "pipeline_failure_rates.csv"), index=False)
strict_cov_tbl.to_csv(os.path.join(OUT_DIR, "strict_valid_rates_ON_COVERED.csv"), index=False)

print("\n=== Coverage (%) ===\n", coverage_tbl)
print("\n=== Pipeline failures (%) ===\n", pipeline_tbl)
print("\n=== Strict-valid (%) among covered ===\n", strict_cov_tbl)


#error type proxies (on usable_strict)
#These are *explainability behavior proxies*, not medical correctness.

FALLBACK_PATTERNS = [
    r"\bfallback\b", r"\bn/a\b", r"\bno advice\b", r"\(fallback advice\)",
]

OVERSIMPLIFY_PATTERNS = [
    r"\bsee a doctor\b", r"\bconsult a doctor\b", r"\bvisit a doctor\b",
    r"\bseek medical attention\b", r"\bgo to the doctor\b",
]

HALLUCINATION_PATTERNS = [
    r"\bmri\b", r"\bct\b", r"\bx-?ray\b",
    r"\bchemotherapy\b", r"\bradiotherapy\b",
    r"\bemergency surgery\b", r"\bicu\b",
]

def semantic_present(term: str, text: str) -> bool:
    term = normalize_text(term)
    text = normalize_text(text)

    if not term or term in ["no change", "multi", ""]:
        return True
    if term in text:
        return True

    t_tokens = set(term.split())
    x_tokens = set(text.split())

    if len(t_tokens) >= 2:
        overlap = len(t_tokens & x_tokens) / max(1, len(t_tokens))
        return overlap >= 0.6

    return False

def detect_error_type(row) -> str:
    #if not usable_strict, classify separately (pipeline/format gating)
    if not bool(row.get("usable_strict", False)):
        return "format_or_pipeline"

    advice = normalize_text(row.get("final_advice", ""))
    term   = normalize_text(row.get("term", ""))
    rep    = normalize_text(row.get("replacement", ""))
    cf_type = (row.get("type") or "").strip()

    #fallback-y text (even if strict_valid passed)
    for p in FALLBACK_PATTERNS:
        if re.search(p, advice):
            return "format_or_fallback"

    #contradiction proxy (useful mostly for negation like antonyms)/ if replacement starts with "no/without", but advice mentions target without negation markers
    if cf_type == "ant" and (rep.startswith("no ") or rep.startswith("without ")):
        target = rep.replace("no ", "").replace("without ", "").strip()
        if target and (target in advice):
            if not any(neg in advice for neg in [
                f"no {target}", f"without {target}", f"not {target}", "denies", "deny"
            ]):
                return "contradiction"

    #missing key concept: advice should mention either original term or replacement (at least loosely)
    if term and len(term) >= 3:
        if (not semantic_present(term, advice)) and (not semantic_present(rep, advice)):
            return "missing_key_concept"

    #hallucination-like escalation proxy
    for p in HALLUCINATION_PATTERNS:
        if re.search(p, advice):
            return "hallucination_like"

    #oversimplification proxy (short generic advice)
    for p in OVERSIMPLIFY_PATTERNS:
        if re.search(p, advice) and len(advice.split()) < 45:
            return "over_simplification"

    return "none"

df_cf["error_type"] = df_cf.apply(detect_error_type, axis=1)
df_cf["is_error"] = ~df_cf["error_type"].isin(["none"])

df_cf.to_csv(os.path.join(OUT_DIR, "df_counterfactuals_all_phases_with_errors.csv"), index=False)
print(f"[INFO] Saved: {OUT_DIR}/df_counterfactuals_all_phases_with_errors.csv")


#summary tables
overall_dist = df_cf["error_type"].value_counts()
by_phase = pd.crosstab(df_cf["phase"], df_cf["error_type"])
by_phase_type = pd.crosstab([df_cf["phase"], df_cf["type"]], df_cf["error_type"])

overall_dist.to_csv(os.path.join(OUT_DIR, "error_distribution_overall.csv"))
by_phase.to_csv(os.path.join(OUT_DIR, "error_distribution_by_phase.csv"))
by_phase_type.to_csv(os.path.join(OUT_DIR, "error_distribution_by_phase_and_type.csv"))

print("\n=== Overall error distribution ===\n", overall_dist)
print("\n=== Error distribution by phase ===\n", by_phase)

#error rates per 100
summary = (
    df_cf.groupby(["phase", "type"])
         .agg(total=("is_error", "size"), errors=("is_error", "sum"))
         .reset_index()
)
summary["error_rate_per_100"] = (summary["errors"] / summary["total"]) * 100
summary.to_csv(os.path.join(OUT_DIR, "error_rates_by_phase_and_type.csv"), index=False)
print("\n=== Error rate per 100 (phase/type) ===\n", summary)

#mean weight by error/non error
w_summary = (
    df_cf[df_cf["usable_strict"]]
    .groupby(["phase", "type", "is_error"])["weight"]
    .mean()
    .reset_index()
)
w_summary.to_csv(os.path.join(OUT_DIR, "mean_weight_by_error_STRICT_ONLY.csv"), index=False)
print("\n=== Mean weight by error (STRICT ONLY) ===\n", w_summary)


#FIGURES
def pivot_metric(df, value_col, index_col="phase", col_col="type"):
    pv = df.pivot_table(index=index_col, columns=col_col, values=value_col, aggfunc="mean").fillna(0.0)
    pv = pv.loc[[p for p in PHASES if p in pv.index]]
    return pv

def plot_two_bars(pv, title, ylabel, filename):
    x = np.arange(len(pv.index))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))

    syn_vals = pv.get("syn", pd.Series([0]*len(x), index=pv.index)).values
    ant_vals = pv.get("ant", pd.Series([0]*len(x), index=pv.index)).values

    ax.bar(x - w/2, syn_vals, w, label="syn")
    ax.bar(x + w/2, ant_vals, w, label="ant")

    ax.set_xticks(x)
    ax.set_xticklabels(pv.index)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    for i, v in enumerate(syn_vals):
        if v > 0:
            ax.text(i - w/2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(ant_vals):
        if v > 0:
            ax.text(i + w/2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    print("[INFO] Saved:", outpath)

#plots
pv_cov = pivot_metric(coverage_tbl, "coverage_rate_pct")
pv_pipe = pivot_metric(pipeline_tbl, "pipeline_failure_rate_pct")
pv_strict = pivot_metric(strict_cov_tbl, "strict_valid_rate_pct")

plot_two_bars(pv_cov, "Coverage Rate by Phase and Type (status = CF_GENERATED)", "Coverage (%)",
             "figure_cov_rate_by_phase_type.png")
plot_two_bars(pv_pipe, "Pipeline Failure Rate by Phase and Type", "Pipeline failures (%)",
             "figure_pipeline_failure_rate_by_phase_type.png")
plot_two_bars(pv_strict, "Strict Validity Rate by Phase and Type (among covered only)", "Strict-valid (%)",
             "figure_strict_valid_rate_on_covered_by_phase_type.png")

#Error types by strategy (syn vs ant) excluding "none"
ct = pd.crosstab(df_cf["type"], df_cf["error_type"])
ct_err = ct.drop(columns=["none"], errors="ignore")

order = [
    "format_or_pipeline",
    "format_or_fallback",
    "missing_key_concept",
    "contradiction",
    "over_simplification",
    "hallucination_like",
]
ct_err = ct_err[[c for c in order if c in ct_err.columns]]

labels = ct_err.columns.tolist()
x = np.arange(len(labels))
w = 0.35

syn_vals = ct_err.loc["syn"].values if "syn" in ct_err.index else np.zeros(len(labels))
ant_vals = ct_err.loc["ant"].values if "ant" in ct_err.index else np.zeros(len(labels))

fig, ax = plt.subplots(figsize=(11, 4))
b1 = ax.bar(x - w/2, syn_vals, w, label="Synonym")
b2 = ax.bar(x + w/2, ant_vals, w, label="Antonym")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylabel("Count")
ax.set_title("Error Types by Counterfactual Strategy (All rows; pipeline separated)")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend()

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{int(h)}",
                ha="center", va="bottom", fontsize=9)

plt.tight_layout()
outpath = os.path.join(FIG_DIR, "figure_error_types_by_strategy.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
print("[INFO] Saved:", outpath)

#error rate per 100 by phase
pivot_rate = (
    summary
    .pivot_table(index="phase", columns="type", values="error_rate_per_100", aggfunc="mean")
    .fillna(0.0)
)
pivot_rate = pivot_rate.loc[[p for p in PHASES if p in pivot_rate.index]]

x = np.arange(len(pivot_rate.index))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 4))
syn_vals = pivot_rate.get("syn", pd.Series([0]*len(x), index=pivot_rate.index)).values
ant_vals = pivot_rate.get("ant", pd.Series([0]*len(x), index=pivot_rate.index)).values

ax.bar(x - w/2, syn_vals, w, label="syn")
ax.bar(x + w/2, ant_vals, w, label="ant")

ax.set_xticks(x)
ax.set_xticklabels(pivot_rate.index)
ax.set_ylabel("Error rate per 100 CFs")
ax.set_title("Normalized Error Rates by Phase and Type")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend()

for i, v in enumerate(syn_vals):
    if v > 0:
        ax.text(i - w/2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(ant_vals):
    if v > 0:
        ax.text(i + w/2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
outpath = os.path.join(FIG_DIR, "figure_error_rate_by_phase_type.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
print("[INFO] Saved:", outpath)

#Weight vs error (STRICT ONLY) — syn
df_syn = df_cf[(df_cf["type"] == "syn") & (df_cf["usable_strict"])].copy()
if len(df_syn) > 0:
    mean_noerr = df_syn[df_syn["is_error"] == False]["weight"].mean()
    mean_err   = df_syn[df_syn["is_error"] == True]["weight"].mean()

    labels2 = ["No Error", "Error"]
    values2 = [mean_noerr, mean_err]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels2, values2)
    ax.set_ylabel("Mean normalized TF-IDF weight")
    ax.set_title("Importance Weight vs Error (SYN, strict-valid only)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, "figure_weight_vs_error_syn_STRICT_ONLY.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    print("[INFO] Saved:", outpath)
else:
    print("[WARN] No SYN strict-valid rows for weight-vs-error plot.")

#Weight vs error (STRICT ONLY) — ant
df_ant = df_cf[(df_cf["type"] == "ant") & (df_cf["usable_strict"])].copy()
if len(df_ant) > 0:
    mean_noerr = df_ant[df_ant["is_error"] == False]["weight"].mean()
    mean_err   = df_ant[df_ant["is_error"] == True]["weight"].mean()

    labels2 = ["No Error", "Error"]
    values2 = [mean_noerr, mean_err]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels2, values2)
    ax.set_ylabel("Mean normalized TF-IDF weight")
    ax.set_title("Importance Weight vs Error (ANT, strict-valid only)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, "figure_weight_vs_error_ant_STRICT_ONLY.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    print("[INFO] Saved:", outpath)
else:
    print("[WARN] No ANT strict-valid rows for weight-vs-error plot.")





#input similarity stats per phase/type (TFIDF + SEQ)
def tfidf_cos(a: str, b: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer().fit([a, b])
        vecs = vect.transform([a, b])
        return float((vecs[0] @ vecs[1].T).toarray()[0][0])
    except Exception:
        return np.nan

def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a), str(b)).ratio()

df_cf["input_tfidf"] = np.nan
df_cf["input_seq"] = np.nan

mask_mod = df_cf["modified_input"].astype(str).str.len() > 0
df_cf.loc[mask_mod, "input_tfidf"] = df_cf.loc[mask_mod].apply(
    lambda r: tfidf_cos(r["orig_input"], r["modified_input"]), axis=1
)
df_cf.loc[mask_mod, "input_seq"] = df_cf.loc[mask_mod].apply(
    lambda r: seq_sim(r["orig_input"], r["modified_input"]), axis=1
)

lex_stats = (
    df_cf.groupby(["phase", "type"])[["input_tfidf", "input_seq"]]
         .agg(["mean", "std"])
)
lex_stats.to_csv(os.path.join(OUT_DIR, "input_similarity_stats_by_phase_and_type.csv"))
print("\n=== Input similarity stats (TFIDF/SEQ) by phase+type ===\n", lex_stats)





final_stats = {
    "coverage_rates": coverage_tbl.to_dict(orient="records"),
    "pipeline_failure_rates": pipeline_tbl.to_dict(orient="records"),
    "strict_valid_rates_on_covered": strict_cov_tbl.to_dict(orient="records"),
    "error_rates_by_phase_and_type": summary.to_dict(orient="records"),
    "error_distribution_overall": overall_dist.to_dict(),
    "error_distribution_by_phase": by_phase.to_dict(),
    "error_distribution_by_phase_and_type": by_phase_type.to_dict(),
}

with open(os.path.join(OUT_DIR, "final_statistics_errors_cache_driven.pkl"), "wb") as f:
    pickle.dump(final_stats, f)

print(f"\n[INFO] Saved: {OUT_DIR}/final_statistics_errors_cache_driven.pkl")
print(f"[INFO] Figures saved in: {FIG_DIR}/")
print(f"[INFO] Tables/CSVs saved in: {OUT_DIR}/")


#METRICS&WEIGHT STATS
#weight stats (from df_cf):
#mean/std/median/p90 for weight and weight_raw


OUT_DIR = "Section9_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

PHASES = ["SYN_ONLY", "ANT_ONLY", "BOTH"]

#evaluation set "STRICT": usable_strict only
FILTER_MODE = "STRICT"

def mean(x):
    x = np.array(list(x), dtype=float)
    return float(np.mean(x)) if len(x) else None

def std(x):
    x = np.array(list(x), dtype=float)
    return float(np.std(x)) if len(x) else None

def median(x):
    x = np.array(list(x), dtype=float)
    return float(np.median(x)) if len(x) else None

def p90(x):
    x = np.array(list(x), dtype=float)
    return float(np.quantile(x, 0.90)) if len(x) else None

def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a), str(b)).ratio()

def tfidf_cos(a: str, b: str) -> float:
    try:
        a = "" if a is None else str(a)
        b = "" if b is None else str(b)
        vect = TfidfVectorizer().fit([a, b])
        vecs = vect.transform([a, b])
        return float((vecs[0] @ vecs[1].T).toarray()[0][0])
    except Exception:
        return np.nan

def pick_eval_df(dfp: pd.DataFrame) -> pd.DataFrame:
    mode = FILTER_MODE.upper()
    if mode == "STRICT":
        return dfp[dfp["usable_strict"]].copy()
    if mode == "COVERED":
        return dfp[(dfp["covered"]) & (dfp["has_advice"])].copy()
    return dfp.copy()

def compute_phase_stats(df_cf: pd.DataFrame, phase: str):
    dfp = df_cf[df_cf["phase"] == phase].copy()
    df_eval = pick_eval_df(dfp)

    #SYN-only input similarity
    df_syn = df_eval[df_eval["type"] == "syn"].copy()
    if len(df_syn) > 0:
        df_syn["in_tfidf"] = df_syn.apply(lambda r: tfidf_cos(r["orig_input"], r["modified_input"]), axis=1)
        df_syn["in_seq"]   = df_syn.apply(lambda r: seq_sim(r["orig_input"], r["modified_input"]), axis=1)

    #output similarity for all rows in eval set
    if len(df_eval) > 0:
        df_eval["out_tfidf"] = df_eval.apply(lambda r: tfidf_cos(r["orig_output"], r["final_advice"]), axis=1)
        df_eval["out_seq"]   = df_eval.apply(lambda r: seq_sim(r["orig_output"], r["final_advice"]), axis=1)

    #weight stats from df_cf eval rows
    w = df_eval["weight"].astype(float).fillna(0.0).clip(lower=0.0).values if "weight" in df_eval.columns else np.array([])
    wr = df_eval["weight_raw"].astype(float).fillna(0.0).clip(lower=0.0).values if "weight_raw" in df_eval.columns else np.array([])

    #split by type within phase
    df_eval_syn = df_eval[df_eval["type"]=="syn"].copy()
    df_eval_ant = df_eval[df_eval["type"]=="ant"].copy()

    stats = {
        "phase": phase,
        "filter_mode": FILTER_MODE.upper(),

        "counts": {
            "df_cf_rows_total": int(len(dfp)),
            "eval_rows_total": int(len(df_eval)),
            "eval_rows_syn": int(len(df_eval_syn)),
            "eval_rows_ant": int(len(df_eval_ant)),
        },

        "weights_from_df_eval": {
            "weight_norm_mean": mean(w),
            "weight_norm_std": std(w),
            "weight_norm_median": median(w),
            "weight_norm_p90": p90(w),

            "weight_raw_mean": mean(wr),
            "weight_raw_std": std(wr),
            "weight_raw_median": median(wr),
            "weight_raw_p90": p90(wr),
        },

        "posthoc_similarity": {
            # SYN input sim (may be None if no syn rows in this phase)
            "syn_in_vs_orig_in_tfidf": {
                "mean": None if len(df_syn)==0 else mean(df_syn["in_tfidf"].dropna().values),
                "std":  None if len(df_syn)==0 else std(df_syn["in_tfidf"].dropna().values),
            },
            "syn_in_vs_orig_in_seq": {
                "mean": None if len(df_syn)==0 else mean(df_syn["in_seq"].dropna().values),
                "std":  None if len(df_syn)==0 else std(df_syn["in_seq"].dropna().values),
            },

            # Output sim for all eval rows
            "out_vs_orig_out_tfidf_all": {
                "mean": None if len(df_eval)==0 else mean(df_eval["out_tfidf"].dropna().values),
                "std":  None if len(df_eval)==0 else std(df_eval["out_tfidf"].dropna().values),
            },
            "out_vs_orig_out_seq_all": {
                "mean": None if len(df_eval)==0 else mean(df_eval["out_seq"].dropna().values),
                "std":  None if len(df_eval)==0 else std(df_eval["out_seq"].dropna().values),
            },

            # (optional) output sim split by type (nice for thesis)
            "out_vs_orig_out_tfidf_syn": {
                "mean": None if len(df_eval_syn)==0 else mean(df_eval_syn.apply(lambda r: tfidf_cos(r["orig_output"], r["final_advice"]), axis=1).dropna().values),
                "std":  None if len(df_eval_syn)==0 else std(df_eval_syn.apply(lambda r: tfidf_cos(r["orig_output"], r["final_advice"]), axis=1).dropna().values),
            },
            "out_vs_orig_out_tfidf_ant": {
                "mean": None if len(df_eval_ant)==0 else mean(df_eval_ant.apply(lambda r: tfidf_cos(r["orig_output"], r["final_advice"]), axis=1).dropna().values),
                "std":  None if len(df_eval_ant)==0 else std(df_eval_ant.apply(lambda r: tfidf_cos(r["orig_output"], r["final_advice"]), axis=1).dropna().values),
            },
        },
    }

    return stats

def run_metrics_statistics(df_cf: pd.DataFrame):
    results = {}
    for ph in PHASES:
        print(f"[INFO] Computing METRICS for phase={ph}  (FILTER_MODE={FILTER_MODE})")
        results[ph] = compute_phase_stats(df_cf, ph)

    #save PKL
    out_pkl = os.path.join(OUT_DIR, "final_statistics_metrics_clean.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"[INFO] Saved: {out_pkl}")

    #save flat CSV
    flat_rows = []
    for ph, r in results.items():
        row = {"phase": ph, "filter_mode": r["filter_mode"]}
        row.update({f"counts__{k}": v for k, v in r["counts"].items()})
        row.update({f"weights__{k}": v for k, v in r["weights_from_df_eval"].items()})
        for k, vv in r["posthoc_similarity"].items():
            row[f"posthoc__{k}__mean"] = vv["mean"]
            row[f"posthoc__{k}__std"]  = vv["std"]
        flat_rows.append(row)

    out_csv = os.path.join(OUT_DIR, "final_statistics_metrics_clean.csv")
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv}")

    return results

def pretty_print_metrics(results: dict):
    for phase, r in results.items():
        print("\n" + "=" * 68)
        print(f"PHASE: {phase} — METRICS (FILTER={r['filter_mode']})")
        print("=" * 68)

        c = r["counts"]
        print(f"Rows: total={c['df_cf_rows_total']} | eval={c['eval_rows_total']} | syn={c['eval_rows_syn']} | ant={c['eval_rows_ant']}")

        w = r["weights_from_df_eval"]
        print("\nWeights (eval rows):")
        print(f"  weight_norm mean={w['weight_norm_mean']:.4f} std={w['weight_norm_std']:.4f} median={w['weight_norm_median']:.4f} p90={w['weight_norm_p90']:.4f}")
        print(f"  weight_raw  mean={w['weight_raw_mean']:.4f}  std={w['weight_raw_std']:.4f}  median={w['weight_raw_median']:.4f}  p90={w['weight_raw_p90']:.4f}")

        m = r["posthoc_similarity"]
        print("\nPost-hoc similarity:")
        for k, v in m.items():
            if v["mean"] is None:
                print(f"  {k}: None")
            else:
                print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")


metrics_results = run_metrics_statistics(df_cf)
pretty_print_metrics(metrics_results)

#FIGURES:METRICS (STRICT/ROBUST)
#SYN_ONLY: syn_out, syn_in  (robust + strict)
#ANT_ONLY: ant_out, ant_in  (robust + strict)
#BOTH: syn_out, syn_in, ant_out, ant_in (robust + strict)
#keys are: cosine, seq, bert, rougeL, meteor
#label: COSINE, SEQ, BERT, ROUGE-L, METEOR

FIG_DIR = "Figures"
os.makedirs(FIG_DIR, exist_ok=True)


AGG = {
    "SYN_ONLY": {
        "ROBUST_OUT": {"syn_out": {"cosine": 0.26141323604194117, "seq": 0.0673950406730045, "bert": 0.7048098798476014, "rougeL": 0.28988690777364223, "meteor": 0.281493108260578}},
        "STRICT_OUT": {"syn_out": {"cosine": 0.26141323604194117, "seq": 0.0673950406730045, "bert": 0.7048098798476014, "rougeL": 0.28988690777364223, "meteor": 0.281493108260578}},
        "ROBUST_IN":  {"syn_in":  {"cosine": 0.9688334313042938,  "seq": 0.9310180753304113,  "bert": 0.980077110637318,  "rougeL": 0.975472220927322,   "meteor": 0.9744927818793131}},
        "STRICT_IN":  {"syn_in":  {"cosine": 0.9670066553924335,  "seq": 0.9272939399623352,  "bert": 0.9793712987619287, "rougeL": 0.9744889999797953,  "meteor": 0.9734366853653814}},
    },

    "ANT_ONLY": {
        "ROBUST_OUT": {"ant_out": {"cosine": 0.07413759272359675, "seq": 0.03228396149679262, "bert": 0.526482044643647,  "rougeL": 0.12226775673798806, "meteor": 0.10662979930325736}},
        "STRICT_OUT": {"ant_out": {"cosine": 0.08136082910944406, "seq": 0.0385207776217485,  "bert": 0.5410268279639158, "rougeL": 0.12487293050160916, "meteor": 0.10754141765211189}},
        "ROBUST_IN":  {"ant_in":  {"cosine": 0.9780195833685006,  "seq": 0.9422659941604424,  "bert": 0.9744834430515766, "rougeL": 0.9848557847809337,  "meteor": 0.9867266524200119}},
        "STRICT_IN":  {"ant_in":  {"cosine": 0.9750148063592711,  "seq": 0.9458327620577763,  "bert": 0.9744253465623567, "rougeL": 0.9828475494137455,  "meteor": 0.9853145861496947}},
    },

    "BOTH": {
        "ROBUST_OUT": {
            "syn_out": {"cosine": 0.07428461769405686, "seq": 0.02978853583946949, "bert": 0.5448374599218369, "rougeL": 0.12050851838764376, "meteor": 0.10323876774161526},
            "ant_out": {"cosine": 0.07413759272359675, "seq": 0.03228396149679262, "bert": 0.526482044643647,  "rougeL": 0.12226775673798806, "meteor": 0.10662979930325736},
        },
        "STRICT_OUT": {
            "syn_out": {"cosine": 0.07607255232396383, "seq": 0.03158259188775588, "bert": 0.569476802502909,  "rougeL": 0.1261307777434881,  "meteor": 0.10770772402149394},
            "ant_out": {"cosine": 0.08136082910944406, "seq": 0.0385207776217485,  "bert": 0.5410268279639158, "rougeL": 0.12487293050160916, "meteor": 0.10754141765211189},
        },
        "ROBUST_IN": {
            "syn_in": {"cosine": 0.9735046237558012, "seq": 0.8771335975965124, "bert": 0.9779747334810404, "rougeL": 0.97960173105067,   "meteor": 0.9800347269171448},
            "ant_in": {"cosine": 0.9780195833685006, "seq": 0.9422659941604424, "bert": 0.9744834430515766, "rougeL": 0.9848557847809337, "meteor": 0.9867266524200119},
        },
        "STRICT_IN": {
            "syn_in": {"cosine": 0.9706373002732785, "seq": 0.8922181379604027, "bert": 0.9797891359175405, "rougeL": 0.9778976953419763, "meteor": 0.9763561485917616},
            "ant_in": {"cosine": 0.9750148063592711, "seq": 0.9458327620577763, "bert": 0.9744253465623567, "rougeL": 0.9828475494137455, "meteor": 0.9853145861496947},
        },
    },
}

LABELS_5 = ["COSINE", "SEQ", "BERT", "ROUGE-L", "METEOR"]
KEYS_5   = ["cosine", "seq", "bert", "rougeL", "meteor"]

LABELS_3 = ["COSINE", "SEQ", "BERT"]
KEYS_3   = ["cosine", "seq", "bert"]

def grouped_bar_strict_vs_robust(title, strict_vals, robust_vals, labels, out_png):
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x - w/2, strict_vals, w, label="STRICT")
    ax.bar(x + w/2, robust_vals, w, label="ROBUST")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for i, v in enumerate(strict_vals):
        ax.text(i - w/2, min(0.99, v + 0.02), f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(robust_vals):
        ax.text(i + w/2, min(0.99, v + 0.02), f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, out_png), dpi=300, bbox_inches="tight")
    plt.show()

def line_chart_multi(title, series_dict, labels, out_png):
    # series_dict: {name: [vals aligned with labels]}
    plt.figure(figsize=(8,4))
    for name, vals in series_dict.items():
        plt.plot(labels, vals, marker="o", label=name)

    plt.ylim(0, 1)
    plt.ylabel("Similarity Score")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()


    for name, vals in series_dict.items():
        for i, v in enumerate(vals):
            plt.text(i, min(0.98, v + 0.03), f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, out_png), dpi=300, bbox_inches="tight")
    plt.show()

def heatmap(title, df: pd.DataFrame, out_png):
    fig, ax = plt.subplots(figsize=(7,4))
    im = ax.imshow(df.values, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i,j]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Comparison")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, out_png), dpi=300, bbox_inches="tight")
    plt.show()


#grouped bars: STRICT vs ROBUST (SYN_ONLY syn_out)
ph = "SYN_ONLY"
strict_vals = [AGG[ph]["STRICT_OUT"]["syn_out"][k] for k in KEYS_5]
robust_vals = [AGG[ph]["ROBUST_OUT"]["syn_out"][k] for k in KEYS_5]
grouped_bar_strict_vs_robust(
    title="SYN_ONLY: syn_out_vs_orig_out (STRICT vs ROBUST)",
    strict_vals=strict_vals,
    robust_vals=robust_vals,
    labels=LABELS_5,
    out_png="fig_metrics_SYN_ONLY_syn_out_strict_vs_robust.png",
)


#grouped bars: STRICT vs ROBUST (ANT_ONLY ant_out)
ph = "ANT_ONLY"
strict_vals = [AGG[ph]["STRICT_OUT"]["ant_out"][k] for k in KEYS_5]
robust_vals = [AGG[ph]["ROBUST_OUT"]["ant_out"][k] for k in KEYS_5]
grouped_bar_strict_vs_robust(
    title="ANT_ONLY: ant_out_vs_orig_out (STRICT vs ROBUST)",
    strict_vals=strict_vals,
    robust_vals=robust_vals,
    labels=LABELS_5,
    out_png="fig_metrics_ANT_ONLY_ant_out_strict_vs_robust.png",
)


#grouped bars: STRICT vs ROBUST (BOTH syn_out + ant_out)
ph = "BOTH"
for comp in ["syn_out", "ant_out"]:
    strict_vals = [AGG[ph]["STRICT_OUT"][comp][k] for k in KEYS_5]
    robust_vals = [AGG[ph]["ROBUST_OUT"][comp][k] for k in KEYS_5]
    grouped_bar_strict_vs_robust(
        title=f"BOTH: {comp}_vs_orig_out (STRICT vs ROBUST)",
        strict_vals=strict_vals,
        robust_vals=robust_vals,
        labels=LABELS_5,
        out_png=f"fig_metrics_BOTH_{comp}_strict_vs_robust.png",
    )


#sensitivity line charts (STRICT): compare IN vs OUT
#SYN_ONLY: syn_out vs syn_in
#ANT_ONLY: ant_out vs ant_in
#BOTH: syn_out vs syn_in vs ant_out vs ant_in
#SYN_ONLY
ph = "SYN_ONLY"
series = {
    "syn_out (STRICT)": [AGG[ph]["STRICT_OUT"]["syn_out"][k] for k in KEYS_3],
    "syn_in  (STRICT)": [AGG[ph]["STRICT_IN"]["syn_in"][k] for k in KEYS_3],
}
line_chart_multi(
    title="SYN_ONLY: STRICT Sensitivity (IN vs OUT)",
    series_dict=series,
    labels=LABELS_3,
    out_png="fig_metrics_SYN_ONLY_strict_sensitivity_in_vs_out.png",
)

#ANT_ONLY
ph = "ANT_ONLY"
series = {
    "ant_out (STRICT)": [AGG[ph]["STRICT_OUT"]["ant_out"][k] for k in KEYS_3],
    "ant_in  (STRICT)": [AGG[ph]["STRICT_IN"]["ant_in"][k] for k in KEYS_3],
}
line_chart_multi(
    title="ANT_ONLY: STRICT Sensitivity (IN vs OUT)",
    series_dict=series,
    labels=LABELS_3,
    out_png="fig_metrics_ANT_ONLY_strict_sensitivity_in_vs_out.png",
)

#BOTH
ph = "BOTH"
series = {
    "syn_out (STRICT)": [AGG[ph]["STRICT_OUT"]["syn_out"][k] for k in KEYS_3],
    "syn_in  (STRICT)": [AGG[ph]["STRICT_IN"]["syn_in"][k] for k in KEYS_3],
    "ant_out (STRICT)": [AGG[ph]["STRICT_OUT"]["ant_out"][k] for k in KEYS_3],
    "ant_in  (STRICT)": [AGG[ph]["STRICT_IN"]["ant_in"][k] for k in KEYS_3],
}
line_chart_multi(
    title="BOTH: STRICT Sensitivity (IN vs OUT)",
    series_dict=series,
    labels=LABELS_3,
    out_png="fig_metrics_BOTH_strict_sensitivity_in_vs_out.png",
)


#heatmaps (STRICT)-(COSINE/SEQ/BERT)
#one heatmap per phase
def make_heat_df(phase: str):
    rows = []
    idx = []
    if phase == "SYN_ONLY":
        rows.append([AGG[phase]["STRICT_OUT"]["syn_out"][k] for k in KEYS_3]); idx.append("syn_out_vs_orig_out")
        rows.append([AGG[phase]["STRICT_IN"]["syn_in"][k] for k in KEYS_3]);   idx.append("syn_in_vs_orig_in")
    elif phase == "ANT_ONLY":
        rows.append([AGG[phase]["STRICT_OUT"]["ant_out"][k] for k in KEYS_3]); idx.append("ant_out_vs_orig_out")
        rows.append([AGG[phase]["STRICT_IN"]["ant_in"][k] for k in KEYS_3]);   idx.append("ant_in_vs_orig_in")
    elif phase == "BOTH":
        rows.append([AGG[phase]["STRICT_OUT"]["syn_out"][k] for k in KEYS_3]); idx.append("syn_out_vs_orig_out")
        rows.append([AGG[phase]["STRICT_IN"]["syn_in"][k] for k in KEYS_3]);   idx.append("syn_in_vs_orig_in")
        rows.append([AGG[phase]["STRICT_OUT"]["ant_out"][k] for k in KEYS_3]); idx.append("ant_out_vs_orig_out")
        rows.append([AGG[phase]["STRICT_IN"]["ant_in"][k] for k in KEYS_3]);   idx.append("ant_in_vs_orig_in")
    return pd.DataFrame(rows, columns=LABELS_3, index=idx)

for ph in ["SYN_ONLY", "ANT_ONLY", "BOTH"]:
    dfh = make_heat_df(ph)
    heatmap(
        title=f"{ph}: STRICT Similarity Heatmap",
        df=dfh,
        out_png=f"fig_metrics_{ph}_strict_heatmap.png",
    )

print(f"[INFO] Saved metrics figures to: {FIG_DIR}/")







