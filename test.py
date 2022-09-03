import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # TODO find a better solution

from simplegrad.engine import Num, relu, log, exp
import torch
import random
import math


# This complex testing is probably overkill, but I have never done any
# randomized testing and thought it would be fun

# params
trials = 10000
min_ops = 1
max_ops = 10
min_start_nums = 1
max_start_nums = 5
min_num = -10
max_num = 10
max_exp = 10 # only allow e^x for x <= max_exp
rtol = 1e-6
atol = 1e-6

verbose = False

random.seed(102)

ops = Num.unary_operations + Num.binary_operations


def main():
    # perform trials
    for _ in range(trials):
        # choose number of ops and start nums
        num_ops = random.randint(min_ops, max_ops)
        num_start_nums = random.randint(min_start_nums, max_start_nums)

        if verbose:
            print(f'num_ops={num_ops}, num_start_nums={num_start_nums}')

        # generate start nums
        start_nums = [random.uniform(min_num, max_num) for _ in range(num_start_nums)]
        nums_simple: list[Num] = [Num(n) for n in start_nums]
        nums_torch: list[torch.Tensor] = [torch.tensor(n, requires_grad=True, dtype=torch.double) for n in start_nums]

        if verbose:
            output = 'start_nums: ['
            for n in nums_simple:
                output += f'{n.value:.3f} '
            print(output[:-1] + ']')

        # do operationions
        ops_log = []
        for i in range(num_ops):
            # pick and operation and operands
            while True:
                op = random.choice(ops)
                num1_index = random.randrange(0, num_start_nums + i)
                num2_index = random.randrange(0, num_start_nums + i)
                if operation_allowed(op, nums_simple[num1_index].value, nums_simple[num2_index].value):
                    break

            # perform operation
            num1_simple = nums_simple[num1_index]
            num2_simple = nums_simple[num2_index]
            num1_torch = nums_torch[num1_index]
            num2_torch = nums_torch[num2_index]
            if op == 'add':
                result_simple = num1_simple + num2_simple
                result_torch = num1_torch + num2_torch
            elif op == 'sub':
                result_simple = num1_simple - num2_simple
                result_torch = num1_torch - num2_torch
            elif op == 'mul':
                result_simple = num1_simple * num2_simple
                result_torch = num1_torch * num2_torch
            elif op == 'div':
                result_simple = num1_simple / num2_simple
                result_torch = num1_torch / num2_torch
            elif op == 'relu':
                result_simple = relu(num1_simple)
                result_torch = torch.relu(num1_torch)
            elif op == 'log':
                result_simple = log(num1_simple)
                result_torch = torch.log(num1_torch)
            else: # exp
                result_simple = exp(num1_simple)
                result_torch = torch.exp(num1_torch)
            nums_simple.append(result_simple)
            nums_torch.append(result_torch)

            if op in Num.unary_operations:
                op_string = f'{op}({num1_simple.value:.3f})'
            else:
                op_string = f'{op}({num1_simple.value:.3f}, {num2_simple.value:.3f})'
            op_string = f'op {i}: {op_string} = {result_simple.value:.3f}'
            ops_log.append(op_string)

            if verbose:
                print(op_string)

            # verify values match
            if not math.isclose(result_simple.value, result_torch, rel_tol=rtol, abs_tol=atol):
                print(f'value match failure. simple={result_simple.value}, torch={result_torch}')
                print(f'\tfor operation "{op}"')
                print(f'\twith inputs: simple=({num1_simple.value}, {num2_simple.value}), torch=({num1_torch}, {num2_torch})')
                print(f'\toperation history: {ops_log}')
                exit()

        # check gradients for each of the result values
        for i, (val_simple, val_torch) in enumerate(zip(nums_simple[num_start_nums:], nums_torch[num_start_nums:])):
            # zero gradients
            for n_simple, n_torch in zip(nums_simple, nums_torch):
                n_simple.zero_grad()
                n_torch.grad = torch.tensor(0, dtype=torch.double)

            # calculate gradients
            val_simple.backward()
            val_torch.backward(retain_graph=True)
            grads_simple: list[float] = []
            for j, (n_simple, n_torch) in enumerate(zip(nums_simple[:num_start_nums], nums_torch[:num_start_nums])):
                # verify match
                if not math.isclose(n_simple.grad, n_torch.grad, rel_tol=rtol, abs_tol=atol):
                    print(f'grad match failure for op {i}, num {j}. simple={n_simple.grad}, torch={n_torch.grad}')
                    print(f'\toperations log: {ops_log}')
                    exit()

                grads_simple.append(n_simple.grad)

            if verbose:
                grads_string_simple = '['
                for g_simple in grads_simple:
                    grads_string_simple += f'{g_simple:.3f} '
                print(f'grads for op {i}: {grads_string_simple[:-1]}]')

    print(f'Successfully ran {trials} trials')


# Ensure op is valid (can't have nonpositive denominator for log or exp greater
# than max_exp
def operation_allowed(op: str, num1: float, num2: float) -> bool:
    if op == 'div' and num2 == 0:
        return False
    elif op == 'log' and num1 <= 0:
        return False
    elif op == 'exp' and num1 > max_exp:
        return False
    return True


if __name__ == '__main__':
    main()

