import argparse
import csv
import json
import os
import pickle
import time
from collections import defaultdict
from multiprocessing.pool import Pool

import spacy
import transformers
from transformers import AutoTokenizer

STOPWORDS = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'also', 'could', 'would'}


class DocumentProcessor:
    def __init__(self, tokenizer, compute_recurring_spans=True, min_span_length=1, max_span_length=10,
                 validate_spans=True, include_sub_clusters=False):
        self.nlp = spacy.load("en_core_web_sm") if compute_recurring_spans else None
        self.tokenizer = tokenizer
        self.compute_recurring_spans = compute_recurring_spans
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.validate_spans = validate_spans
        self.include_sub_clusters = include_sub_clusters

    def _find_all_spans_in_document(self, doc_psgs):
        output = defaultdict(list)
        for psg in doc_psgs:
            psg_idx, psg_tokens = psg
            tokens_txt = [t.text.lower() for t in psg_tokens]
            for i in range(len(psg_tokens)):
                for j in range(i + self.min_span_length, min(i + self.max_span_length, len(psg_tokens))):
                    length = j - i
                    if self.validate_spans:
                        if (length == 1 and DocumentProcessor.validate_length_one_span(psg_tokens[i])) or (
                                length > 1 and DocumentProcessor.validate_ngram(psg_tokens, i, length)):
                            span_str = " ".join(tokens_txt[i: j])
                            output[span_str].append((psg_idx, i, j))
        return output

    @staticmethod
    def validate_ngram(tokens, start_index, length):
        if any((not tokens[idx].is_alpha) and (not tokens[idx].is_digit) for idx in
               range(start_index, start_index + length)):
            return False

        # We filter out n-grams that are all stopwords, or that begin or end with stop words (e.g. "in the", "with my", ...)
        if any(tokens[idx].text.lower() not in STOPWORDS for idx in range(start_index, start_index + length)) and \
                tokens[start_index].text.lower() not in STOPWORDS and tokens[
            start_index + length - 1].text.lower() not in STOPWORDS:
            return True

        # TODO: Consider validating that the recurring span is not contained in the title (and vice versa)
        # span_lower = span.lower()
        # title_lower = title.lower()
        # if span_lower in title_lower or title_lower in span_lower:
        #     return False
        return False

    @staticmethod
    def validate_length_one_span(token):
        return token.text[0].isupper() or (len(token.text) == 4 and token.is_digit)

    @staticmethod
    def _filter_sub_clusters(recurring_spans):
        output = []
        span_txts = recurring_spans.keys()
        span_txts = sorted(span_txts, key=lambda x: len(x))
        for idx, span_txt in enumerate(span_txts):
            locations = recurring_spans[span_txt]
            is_sub_span = False
            for larger_span_txt in span_txts[idx + 1:]:
                if span_txt in larger_span_txt:
                    larger_locations = recurring_spans[larger_span_txt]
                    if len(locations) != len(larger_locations):
                        continue
                    is_different = False
                    for i in range(len(locations)):
                        psg_idx, start, end = locations[i]
                        larger_psg_idx, larger_start, larger_end = larger_locations[i]
                        if not (psg_idx == larger_psg_idx and start >= larger_start and end <= larger_end):
                            is_different = True
                            break
                    if not is_different:
                        is_sub_span = True
                        break
            if not is_sub_span:
                output.append((span_txt, locations))
        return output

    def _find_recurring_spans_in_documents(self, doc_psgs):
        """
        This function gets a list of spacy-tokenized passages and returns the list of recurring spans that appear in
        more than one passage
        Returns: A list of tuples, each representing a recurring span and has two items:
        * A string representing the lower-cased version of the recurring span
        * A list of it occurrences, each represented with a three-item tuple: (psg_index, span_start, span_end)
        """
        # first we get all spans with length >= min_length and validated (if wanted)
        spans_txts_to_locations = self._find_all_spans_in_document(doc_psgs)
        # now we filter out the spans that aren't recurring (or are recurring, but only in one passage)
        recurring = {}
        for span_txt, locations in spans_txts_to_locations.items():
            if len(locations) > 1:
                first_occurrence = locations[0][0]
                # check if span occurs in more than one passage
                for location in locations[1:]:
                    if location[0] != first_occurrence:
                        recurring[span_txt] = locations
                        break
        if self.include_sub_clusters:
            return recurring
        # else, filter out sub_clusters
        output = self._filter_sub_clusters(recurring)
        return output

    def _encode_and_convert_span_indices(self, spacy_tokenized_psgs, title, recurring_spans):
        encoded_psgs = {}
        old_to_new_indices = {}
        encoded_title = self.tokenizer.encode(title, add_special_tokens=False)
        for psg_id, psg in spacy_tokenized_psgs:
            encoded_psg = []
            indices_map = []
            for token in psg:
                new_idx = len(encoded_psg)
                indices_map.append(new_idx)
                encoded_psg.extend(self.tokenizer.encode(token.text, add_special_tokens=False))
            encoded_psgs[psg_id] = (encoded_title, encoded_psg)
            old_to_new_indices[psg_id] = indices_map

        new_recurring_spans = []
        for span_str, span_occurrences in recurring_spans:
            new_span_occurrences = [
                (psg_index, old_to_new_indices[psg_index][span_start], old_to_new_indices[psg_index][span_end])
                for psg_index, span_start, span_end in span_occurrences]
            new_recurring_spans.append((span_str, new_span_occurrences))

        return encoded_psgs, new_recurring_spans

    def _get_candidates_for_recurring_spans(self, psgs):
        """
        This function removes articles with less than a given number of passages.
        In addition, it removes the last passage because it also contains the prefix of the article (which is problematic
        for recurring span identification)
        """
        psgs = psgs[:-1]

        if len(psgs) < 3:
            return None
        return psgs

    def _postprocess_recurring_span_examples(self, recurring_spans, all_candidate_psgs, title):
        new_recurring_spans = []
        for span_str, span_occurrences in recurring_spans:
            positive_ctxs = set([psg_index for psg_index, _, __ in span_occurrences])
            if len(positive_ctxs) < 2:
                continue
            negative_ctxs = list(set([psg_index for psg_index, _ in all_candidate_psgs]) - positive_ctxs)
            if len(negative_ctxs) < 1:
                continue
            new_recurring_spans.append({
                "span": span_str,
                "title": title,
                "positive": span_occurrences,
                "negative_ctxs": negative_ctxs
            })
        return new_recurring_spans

    def process_document(self, psgs, title):
        """
        This function gets a list of string corresponding to passages.
        It tokenizes them and finds clusters of recurring spans across the passages
        """
        if self.compute_recurring_spans:
            tokenized_psgs = [(psg_id, self.nlp(psg_txt)) for psg_id, psg_txt in psgs]
            psgs_for_recurring_spans = self._get_candidates_for_recurring_spans(tokenized_psgs)
            if psgs_for_recurring_spans is not None:
                recurring_spans = self._find_recurring_spans_in_documents(psgs_for_recurring_spans)
            else:
                recurring_spans = []
            encoded_psgs, recurring_spans = self._encode_and_convert_span_indices(tokenized_psgs, title, recurring_spans)
            recurring_spans = self._postprocess_recurring_span_examples(recurring_spans, psgs_for_recurring_spans, title)
            return encoded_psgs, recurring_spans
        else:
            encoded_psgs = {}
            encoded_title = self.tokenizer.encode(title, add_special_tokens=False)
            for psg_id, psg_txt in psgs:
                encoded_psg_txt = self.tokenizer.encode(psg_txt, add_special_tokens=False)
                encoded_psgs[psg_id] = (encoded_title, encoded_psg_txt)
            return encoded_psgs, []


def load_wiki_dump(dump_path):
    # psg_to_article = {}  # psg id -> title
    article_to_psgs = defaultdict(list)  # title -> List of psgs
    print(f"Loading wiki dump from {dump_path}")
    start_time = time.time()
    num_psgs = 0
    with open(dump_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for psg_id, psg_txt, title in reader:
            if psg_id == "id":
                continue
            num_psgs += 1
            psg_id = int(psg_id)
            # psg_to_article[psg_id] = title
            article_to_psgs[title].append((psg_id, psg_txt))
    end_time = time.time()
    print(f"Done! Took {(end_time - start_time) / 60:.1f} minutes")
    return article_to_psgs, num_psgs  # , psg_to_article


def preprocess_shard(shard_idx, args, article_to_psgs, tokenizer, tokenized_psgs_output_path,
                     compute_recurring_spans=True, recurring_span_output_path=None, recurring_span_output_pickle=True):
    all_encoded_psgs = {}
    all_recurring_spans = []
    processor = DocumentProcessor(tokenizer,
                                  compute_recurring_spans=compute_recurring_spans,
                                  max_span_length=args.max_span_length,
                                  min_span_length=args.min_span_length,
                                  validate_spans=True,
                                  include_sub_clusters=False)
    start_time = time.time()
    for i, (title, psgs) in enumerate(article_to_psgs):
        if shard_idx == 0 and i > 0 and i % 1000 == 0:
            minutes = (time.time() - start_time) / 60
            print(f"Finished {i} out of {len(article_to_psgs)} articles in shard 0. Took {minutes:0.1f} minutes")
        encoded_psgs, recurring_spans = processor.process_document(psgs, title)
        all_encoded_psgs.update(encoded_psgs)
        all_recurring_spans.extend(recurring_spans)

    minutes = (time.time() - start_time) / 60
    print(f"Finished processing {len(encoded_psgs)} in shard {shard_idx}. Took {minutes:0.1f} minutes")

    # return all_tokenized_psgs, all_recurring_spans
    with open(tokenized_psgs_output_path, "wb") as f:
        pickle.dump(all_encoded_psgs, f)
    print(f"Finished writing tokenized passages in shard {shard_idx}. File: {tokenized_psgs_output_path}")
    if compute_recurring_spans:
        with open(recurring_span_output_path, "wb" if recurring_span_output_pickle else "w") as f:
            if recurring_span_output_pickle:
                pickle.dump(all_recurring_spans, f)
            else:
                for recurring_span in all_recurring_spans:
                    f.write(json.dumps(recurring_span))
                    f.write("\n")
        print(f"Finished writing recurring spans in shard {shard_idx}. File: {recurring_span_output_path}")


def split_articles_to_shards(articles, num_psgs_in_shard, num_shards, num_psgs):
    all_shards = []
    curr_article_idx = 0
    article_titles = list(articles.keys())
    psgs_counter = 0
    for _ in range(num_shards):
        curr_shard_articles = []
        num_psgs_in_curr_shard = 0
        while num_psgs_in_curr_shard < num_psgs_in_shard and curr_article_idx < len(article_titles):
            article_title = article_titles[curr_article_idx]
            article_psgs = articles[article_title]
            curr_shard_articles.append((article_title, article_psgs))

            num_psgs_in_curr_shard += len(article_psgs)
            psgs_counter += len(article_psgs)
            curr_article_idx += 1
        all_shards.append(curr_shard_articles)
    assert psgs_counter == num_psgs
    return all_shards


def create_params_for_multiprocessing(args, articles, num_psgs, tokenizer):
    params = []
    num_shards = args.num_processes
    num_psgs_in_shard = num_psgs // num_shards + 1
    shards = split_articles_to_shards(articles, num_psgs_in_shard, num_shards, num_psgs)
    compute_recurring_spans = args.compute_recurring_spans
    for shard_idx, shard_articles in enumerate(shards):
        tokenized_file = os.path.join(args.output_dir, f"tokenized_{shard_idx}.pkl")
        recurring_span_file = os.path.join(args.output_dir, f"recurring_{shard_idx}.pkl") \
            if compute_recurring_spans else None
        params.append((shard_idx, args, shard_articles, tokenizer, tokenized_file, compute_recurring_spans, recurring_span_file))
    print(f"Finished splitting {len(articles)} articles to {args.num_processes} processes")
    return params


def main(args):
    os.makedirs(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    transformers.logging.set_verbosity_error()

    article_to_psgs, num_psgs = load_wiki_dump(args.corpus_path)
    # create_examples(args, article_to_psgs, tokenizer, tokenized_file, recurring_span_file)
    params = create_params_for_multiprocessing(args, article_to_psgs, num_psgs, tokenizer)
    with Pool(args.num_processes if args.num_processes else None) as p:
        p.starmap(preprocess_shard, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--compute_recurring_spans", action="store_true")
    parser.add_argument("--min_span_length", type=int, default=1)
    parser.add_argument("--max_span_length", type=int, default=10)
    parser.add_argument("--num_processes", type=int, default=64)

    args = parser.parse_args()
    main(args)
