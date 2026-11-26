"""Retrieve annotated AVeriTeC evidence and surface exact published references."""

import math
import re
from collections import Counter
from pathlib import Path
import json


def read_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records or any(
        not row.get("claim") or not row.get("label") for row in records
    ):
        raise ValueError("Expected labeled AVeriTeC JSON records")
    return records


def claim_key(text: str) -> str:
    return " ".join(text.strip().casefold().split()).rstrip(".?!")


def terms(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.casefold()) if len(token) > 2 or token.isdigit()}


def evidence(record: dict) -> list[dict]:
    return [
        {"url": answer["source_url"], "question": question["question"],
         "annotation": " ".join(filter(None, (answer.get("answer"), answer.get("boolean_explanation"))))}
        for question in record.get("questions", [])
        for answer in question.get("answers", [])
        if (answer.get("source_url") or "").startswith(("https://", "http://"))
        and answer.get("answer")
    ]


class EvidenceIndex:
    def __init__(self, records: list[dict]):
        self.records = records
        self.matches = {}
        self.docs = []
        distinct = set()
        for record in records:
            self.matches.setdefault(claim_key(record["claim"]), []).append(record)
            for item in evidence(record):
                key = (item["url"], item["question"], item["annotation"])
                if key not in distinct:
                    self.docs.append((item, terms(item["question"] + " " + item["annotation"])))
                    distinct.add(key)
        frequency = Counter(word for _, tokens in self.docs for word in tokens)
        self.weights = {word: math.log1p(len(self.docs) / count) for word, count in frequency.items()}

    def retrieve(self, claim: str, limit: int = 5) -> list[dict]:
        query = terms(claim)
        ranked = sorted(
            ((sum(self.weights[word] for word in sorted(query & tokens)) / math.sqrt(len(tokens)),
              i, item) for i, (item, tokens) in enumerate(self.docs) if query & tokens),
            key=lambda result: (-result[0], result[1]),
        )
        result, urls = [], set()
        for _, _, item in ranked:
            if item["url"] not in urls:
                result.append(item)
                urls.add(item["url"])
            if len(result) == limit:
                break
        return result

    def report(self, claim: str) -> dict:
        hits = self.matches.get(claim_key(claim), [])
        if len(hits) == 1:
            hit = hits[0]
            return {"claim": claim, "status": "published_reference", "verdict": hit["label"],
                    "claim_date": hit.get("claim_date"), "reference_article": hit.get("fact_checking_article"),
                    "evidence": evidence(hit)}
        return {"claim": claim, "status": "ambiguous_reference" if hits else "needs_review",
                "verdict": None, "candidate_evidence": self.retrieve(claim)}

    def benchmark(self) -> dict:
        eligible, hit = 0, 0
        for row in self.records:
            gold = {item["url"] for item in evidence(row)}
            if gold:
                eligible += 1
                hit += bool(gold & {item["url"] for item in self.retrieve(row["claim"])})
        if not eligible:
            raise ValueError("The reference set has no annotated evidence URLs")
        return {"records": len(self.records), "claims_with_evidence_urls": eligible,
                "hit_at_5_count": hit, "hit_at_5": hit / eligible,
                "candidate_pool": "Annotated question-answer snippets from this reference split",
                "measures": "Evidence URL retrieval within the provided pool"}
