import logging
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def create_query_transformation(args, tokenizer):
    transformation = args.query_transformation
    if transformation == "mask":
        return MaskSpanQueryTransformation(tokenizer, question_id=103)
    elif transformation == "prefix":
        return PrefixQueryTransformation(tokenizer, max_window_size=args.max_window_size)
    elif transformation == "random":
        return RandomWindowQueryTransformation(
            tokenizer,
            min_window_size=args.min_window_size,
            max_window_size=args.max_window_size,
            keep_answer_prob=args.keep_answer_prob,
            random_window=(not args.prefix_window),
            replace_with_question_token=args.replace_with_question_token,
            question_token_id=103,
        )
    assert False, f"Unknown query transformation: {transformation}"


class QueryTransformation:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.example_counter = 0

    def transform_query(self, question_psg_tokens, answer_spans):
        question_tokens = self._transform_query(question_psg_tokens, answer_spans)
        if self.example_counter < 20:
            psg_text = self.tokenizer.decode(question_psg_tokens)
            first_span_start, first_span_end = answer_spans[0]
            answer_text = self.tokenizer.decode(question_psg_tokens[first_span_start:first_span_end])
            question_text = self.tokenizer.decode(question_tokens)

            logger.info("### Passage: #####")
            logger.info(psg_text)
            logger.info("### Answer Span: ###")
            logger.info(answer_text)
            logger.info(f"{answer_spans}")
            logger.info("### Question: #####")
            logger.info(question_text)
            logger.info("*" * 40)

        self.example_counter += 1
        return question_tokens

    def _transform_query(self, question_psg_tokens, answer_spans):
        raise NotImplementedError


class MaskSpanQueryTransformation(QueryTransformation):
    """
    This transformation simply replaces all the given spans with a special token 'question_id' (defaults to '[MASK]')
    """
    def __init__(self, tokenizer, question_id=103):
        super(MaskSpanQueryTransformation, self).__init__(tokenizer)
        self.question_id = question_id

    def _transform_query(self, question_psg_tokens, answer_spans):
        question_tokens = []
        curr_index = 0
        for start_idx, end_idx in answer_spans:
            question_tokens.extend(question_psg_tokens[curr_index:start_idx])
            curr_index = end_idx
        question_tokens.extend(question_psg_tokens[curr_index:])
        return question_tokens


class PrefixQueryTransformation(QueryTransformation):
    """
    This transformation keeps only the prefix before the first span
    """
    def __init__(self, tokenizer, max_window_size=None, minimal_length=5):
        super(PrefixQueryTransformation, self).__init__(tokenizer)
        self.max_window_size = max_window_size
        self.minimal_length = minimal_length

    def _transform_query(self, question_psg_tokens, answer_spans):
        question_tokens = []
        start_idx, end_idx = answer_spans[0]
        if self.max_window_size is None:
            start_question_idx = 0
        else:
            start_question_idx = max(0, start_idx - self.max_window_size)
        question_tokens.extend(question_psg_tokens[start_question_idx:start_idx])

        if len(question_tokens) < self.minimal_length:
            question_tokens.extend(question_psg_tokens[end_idx:end_idx + self.minimal_length])

        return question_tokens


class RandomWindowQueryTransformation(QueryTransformation):
    """
    A random window surrounding one of the answer spans
    """
    def __init__(self, tokenizer, min_window_size=10, max_window_size=30, keep_answer_prob=0.0, random_window=True,
                 replace_with_question_token=False, question_token_id=103):
        super(RandomWindowQueryTransformation, self).__init__(tokenizer)

        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.keep_answer_prob = keep_answer_prob
        self.random_window = random_window
        self.replace_with_question_token = replace_with_question_token
        self.question_token_id = question_token_id

    def _transform_query(self, question_psg_tokens, answer_spans):
        # Choosing one of the spans in random
        span_start_idx, span_end_idx = random.choice(answer_spans)
        span_len = span_end_idx - span_start_idx
        # Choosing query length
        query_length = random.randint(self.min_window_size, self.max_window_size)

        # Defining the optional range for start index of the query
        if self.random_window:
            min_idx = max(0, span_start_idx - query_length)
            max_idx = min(span_end_idx, len(question_psg_tokens) - query_length - span_len)
            query_start_idx = random.randint(min_idx, max_idx)
        else:
            query_start_idx = max(0, span_start_idx - query_length)

        keep_answer = False
        if self.keep_answer_prob > 0.0:
            keep_answer = (random.random() < self.keep_answer_prob)

        if keep_answer:
            question_tokens = question_psg_tokens[query_start_idx:query_start_idx+query_length]
        else:
            question_tokens = []
            question_tokens.extend(question_psg_tokens[query_start_idx:span_start_idx])
            if self.replace_with_question_token:
                question_tokens.append(self.question_token_id)
            num_tokens_remained = query_length - len(question_tokens)
            if num_tokens_remained > 0:
                question_tokens.extend(question_psg_tokens[span_end_idx:span_end_idx+num_tokens_remained])

        assert query_length <= len(question_tokens) <= (query_length + 1), f"{query_length}, {len(question_tokens)}"
        return question_tokens



