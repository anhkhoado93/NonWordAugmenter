import re

DETOKENIZER_REGEXS = [
	(re.compile(r'\s([.,:;?!%]+)([ \'"`])'), r'\1\2'), # End of sentence
	(re.compile(r'\s([.,:;?!%]+)$'), r'\1'), # End of sentence
	(re.compile(r'\s([\[\(\{\<])\s'), r' \g<1>'), # Left bracket
	(re.compile(r'\s([\]\)\}\>])\s'), r'\g<1> '),# right bracket
    (re.compile(r'\s([\.?!])\s*([\.?!]+)'), r'\1\2'),
    (re.compile(r'([\[\(\{\"\'])\s+(.*)\s+([\]\)\}\"\'])'), r'\1\2\3'),
    (re.compile(r'([,?!])(?!([\s.]))(\w)'), r'\1 \3'),
    (re.compile(r'(\d)\s+([,.;])'), r'\1\2'),
    (re.compile(r'\. \.'), r'..'),
    (re.compile(r',\s+,'), r',,'),
    (re.compile(r'([\(\[\{])\s+([\(\[\{])'), r'\1\2'),
    (re.compile(r'([\]\)\}])\s+([\]\)\}])'), r'\1\2')
]

TOKENIZER_REGEX = re.compile(r'(\W)')

def tokenize(text):
    tokens = TOKENIZER_REGEX.split(text)
    return [t for t in tokens if len(t.strip()) > 0]

def normalize_punctuation(data):
    if isinstance(data, str):
        tokens = tokenize(data)
    else:
        tokens = data
    text = ' '.join(tokens)
    for regex, sub in DETOKENIZER_REGEXS:
        text = regex.sub(sub, text)
    return text.strip()