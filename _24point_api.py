OPERATORS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '×': lambda a, b: a * b,
    '*': lambda a, b: a * b,
    '÷': lambda a, b: a / b,
    '/': lambda a, b: a / b
}
EPSILON = 1e-9
MAX_RAW_EXPRESSION_LENGTH = 20
def evaluate_parsed_expression(parsed_expr_list):
    if not parsed_expr_list: return None
    try:
        current_result = float(parsed_expr_list[0])
        for i in range(1, len(parsed_expr_list), 2):
            operator_symbol = parsed_expr_list[i]
            next_number = float(parsed_expr_list[i + 1])
            operation_func = OPERATORS.get(operator_symbol)
            if not operation_func: return None
            if operator_symbol in ('÷', '/') and abs(next_number) < EPSILON:
                return None
            current_result = operation_func(current_result, next_number)
        return current_result
    except (ValueError, IndexError, TypeError):
        return None
def parse_raw_tokens(raw_token_list):
    parsed_list = []
    current_digits_for_number = []
    def finalize_number():
        nonlocal current_digits_for_number
        if not current_digits_for_number: return False
        if current_digits_for_number[0] == 0 and len(current_digits_for_number) > 1:
            return False
        number = 0
        for digit in current_digits_for_number:
            number = number * 10 + digit
        parsed_list.append(number)
        current_digits_for_number = []
        return True
    if raw_token_list and not isinstance(raw_token_list[0], int):
        return None, False
    for token in raw_token_list:
        if isinstance(token, int):
            current_digits_for_number.append(token)
        elif isinstance(token, str) and token in OPERATORS:
            if not finalize_number(): return None, False
            if parsed_list and isinstance(parsed_list[-1], str): return None, False
            parsed_list.append(token)
        else:
            return None, False
    if not finalize_number(): return None, False
    if not parsed_list or len(parsed_list) % 2 == 0:
        return None, False
    for i, item in enumerate(parsed_list):
        is_number_position = (i % 2 == 0)
        if is_number_position and not isinstance(item, (int, float)): return None, False
        if not is_number_position and (
                not isinstance(item, str) or item not in OPERATORS): return None, False
    return parsed_list, True
def solve_recursively(target_op_symbol,
                      all_digits_with_indices,
                      current_raw_tokens,
                      used_digit_indices,
                      max_len):
    if current_raw_tokens and isinstance(current_raw_tokens[-1], int):
        parsed_expression, is_valid = parse_raw_tokens(current_raw_tokens)
        if is_valid:
            result = evaluate_parsed_expression(parsed_expression)
            if result is not None and abs(result - 24) < EPSILON:
                return " ".join(map(str, parsed_expression))
    if len(current_raw_tokens) >= max_len:
        return None
    for digit_value, original_idx in all_digits_with_indices:
        if original_idx not in used_digit_indices:
            solution = solve_recursively(target_op_symbol, all_digits_with_indices,
                                         current_raw_tokens + [digit_value],
                                         used_digit_indices | {original_idx},
                                         max_len)
            if solution: return solution
    if current_raw_tokens and isinstance(current_raw_tokens[-1], int) and used_digit_indices:
        solution = solve_recursively(target_op_symbol, all_digits_with_indices,
                                     current_raw_tokens + [target_op_symbol],
                                     used_digit_indices,
                                     max_len)
        if solution: return solution
    return None
def find_simple_24_formula(target_operator_symbol, four_digits_str):
    if not (isinstance(target_operator_symbol, str) and
            target_operator_symbol in OPERATORS and
            isinstance(four_digits_str, str) and
            len(four_digits_str) == 4 and
            four_digits_str.isdigit()):
        return None
    digits_with_indices = [(int(digit_char), idx) for idx, digit_char in enumerate(four_digits_str)]
    for initial_digit_value, initial_digit_original_idx in digits_with_indices:
        solution_string = solve_recursively(
            target_operator_symbol,
            digits_with_indices,
            [initial_digit_value],
            {initial_digit_original_idx},
            MAX_RAW_EXPRESSION_LENGTH
        )
        if solution_string:
            return solution_string
    return None