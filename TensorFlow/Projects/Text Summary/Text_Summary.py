import spacy
from string import punctuation
from heapq import nlargest

def summarizer(rawdocs):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(rawdocs)
    stop_words = nlp.Defaults.stop_words
    dir(nlp)
    tokens = [token.text for token in doc]
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stop_words and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    max_freq = max(word_freq.values())

    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    sent_tokens = [sent for sent in doc.sents]

    sent_Scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_Scores.keys():
                    sent_Scores[sent] = word_freq[word.text]
                else:
                    sent_Scores[sent] += word_freq[word.text]

    select_len = int(len(sent_tokens) * 0.3)

    summary = nlargest(select_len, sent_Scores, key=sent_Scores.get)
    final_summary = [word.text for word in summary]
    summary = " ".join(final_summary)
    # print(text)
    # print(summary)
    # print("Length of original text", len(text.split(' ')))
    # print("Length of summary text", len(summary.split(' ')))

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))




