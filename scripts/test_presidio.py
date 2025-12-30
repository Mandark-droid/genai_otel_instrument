from presidio_analyzer import AnalyzerEngine

a = AnalyzerEngine()
text = "Patient DOB: 01/15/1980, Med License: MD123456"

# Test with all recognizers and low threshold
print("Test 1: All entities, threshold=0.0")
results = a.analyze(text=text, language="en", score_threshold=0.0)
print(f"Results count: {len(results)}")
for r in results:
    detected_text = text[r.start : r.end]
    print(f'  - {r.entity_type}: score={r.score:.2f}, text="{detected_text}"')

# Test with specific entities
print("\nTest 2: DATE_TIME and MEDICAL_LICENSE, threshold=0.0")
results2 = a.analyze(
    text=text, language="en", entities=["DATE_TIME", "MEDICAL_LICENSE"], score_threshold=0.0
)
print(f"Results count: {len(results2)}")
for r in results2:
    detected_text = text[r.start : r.end]
    print(f'  - {r.entity_type}: score={r.score:.2f}, text="{detected_text}"')

# Test with a different text that's more obvious
print("\nTest 3: More obvious PII")
text2 = "My email is john@example.com and my phone is 555-1234"
results3 = a.analyze(text=text2, language="en", score_threshold=0.0)
print(f"Results count: {len(results3)}")
for r in results3:
    detected_text = text2[r.start : r.end]
    print(f'  - {r.entity_type}: score={r.score:.2f}, text="{detected_text}"')
