import json
import glob
from sacrebleu.metrics import BLEU

def compute_sentence_bleu(jsonl_file):
    bleu = BLEU()
    outputs = []
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # first, gather the data from each line in the file
            data = json.loads(line)
            # get the translation from that line, which could be labeled as any of the following
            translation = (data.get('direct_translation') or data.get('self-CoT-translation') or data.get('teacher-CoT-translation') or  data.get('teacher-Synthesized-CoT-translation'))
            # get the references in order to compute sentence-level BLEU
            references = []
            references.append(data['reference'])
            references.append(data['reference2'])
            # compute setence-level BLEU --> got the method sentence_score from the documentation provided on github
            score = bleu.sentence_score(translation, references)
            # output in the format from https://huggingface.co/spaces/evaluate-metric/sacrebleu
            result = {'score': score.score, 'counts': score.counts, 'totals': score.totals, 'precisions': score.precisions, 'bp': score.bp, 'sys_len': score.sys_len, 'ref_len': score.ref_len}
            # add this to the list
            outputs.append(result)
    return outputs


def mean_score(outputs):
    if not outputs:
        return None
    # get the score from the BLEU performed on each line
    scores = [i['score'] for i in outputs]
    # get the mean of the scores in order to analyze which method performs the best
    return {'mean_bleu_score': sum(scores) / len(scores)}


def main():
    jsonl_files = glob.glob('*.jsonl')
    # do not go through the mean or scores files that are added
    jsonl_files = [f for f in jsonl_files if not(f.endswith('_bleu_scores.jsonl') or f.endswith('_bleu_mean.jsonl'))]
    for jsonl_file in jsonl_files:
        # get the outputs for the sentence-level BLEUs
        outputs = compute_sentence_bleu(jsonl_file)
        # add each result from the outputs into a jsonl file
        scores_file = jsonl_file.replace('.jsonl', '_bleu_scores.jsonl')
        with open(scores_file, 'w') as f:
            for result in outputs:
                f.write(json.dumps(result) + '\n')
        # get the mean for the BLEU per method
        mean = mean_score(outputs)
        # add the mean into another jsonl file
        mean_file = jsonl_file.replace('.jsonl', '_bleu_mean.jsonl')
        with open(mean_file, 'w') as f:
            f.write(json.dumps(mean))


if __name__ == "__main__":
    main()