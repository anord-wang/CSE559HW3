import numpy as np

BLOSUM62_name = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11,
                 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'B': 20, 'Z': 21, 'X': 22,
                 '-': 23}
BLOSUM62_value = [
    ['4', '-1', '-2', '-2', '0', '-1', '-1', '0', '-2', '-1', '-1', '-1', '-1', '-2', '-1', '1', '0', '-3', '-2', '0',
     '-2', '-1', '0', '-4'],
    ['-1', '5', '0', '-2', '-3', '1', '0', '-2', '0', '-3', '-2', '2', '-1', '-3', '-2', '-1', '-1', '-3', '-2', '-3',
     '-1', '0', '-1', '-4'],
    ['-2', '0', '6', '1', '-3', '0', '0', '0', '1', '-3', '-3', '0', '-2', '-3', '-2', '1', '0', '-4', '-2', '-3', '3',
     '0', '-1', '-4'],
    ['-2', '-2', '1', '6', '-3', '0', '2', '-1', '-1', '-3', '-4', '-1', '-3', '-3', '-1', '0', '-1', '-4', '-3', '-3',
     '4', '1', '-1', '-4'],
    ['0', '-3', '-3', '-3', '9', '-3', '-4', '-3', '-3', '-1', '-1', '-3', '-1', '-2', '-3', '-1', '-1', '-2', '-2',
     '-1', '-3', '-3', '-2', '-4'],
    ['-1', '1', '0', '0', '-3', '5', '2', '-2', '0', '-3', '-2', '1', '0', '-3', '-1', '0', '-1', '-2', '-1', '-2', '0',
     '3', '-1', '-4'],
    ['-1', '0', '0', '2', '-4', '2', '5', '-2', '0', '-3', '-3', '1', '-2', '-3', '-1', '0', '-1', '-3', '-2', '-2',
     '1', '4', '-1', '-4'],
    ['0', '-2', '0', '-1', '-3', '-2', '-2', '6', '-2', '-4', '-4', '-2', '-3', '-3', '-2', '0', '-2', '-2', '-3', '-3',
     '-1', '-2', '-1', '-4'],
    ['-2', '0', '1', '-1', '-3', '0', '0', '-2', '8', '-3', '-3', '-1', '-2', '-1', '-2', '-1', '-2', '-2', '2', '-3',
     '0', '0', '-1', '-4'],
    ['-1', '-3', '-3', '-3', '-1', '-3', '-3', '-4', '-3', '4', '2', '-3', '1', '0', '-3', '-2', '-1', '-3', '-1', '3',
     '-3', '-3', '-1', '-4'],
    ['-1', '-2', '-3', '-4', '-1', '-2', '-3', '-4', '-3', '2', '4', '-2', '2', '0', '-3', '-2', '-1', '-2', '-1', '1',
     '-4', '-3', '-1', '-4'],
    ['-1', '2', '0', '-1', '-3', '1', '1', '-2', '-1', '-3', '-2', '5', '-1', '-3', '-1', '0', '-1', '-3', '-2', '-2',
     '0', '1', '-1', '-4'],
    ['-1', '-1', '-2', '-3', '-1', '0', '-2', '-3', '-2', '1', '2', '-1', '5', '0', '-2', '-1', '-1', '-1', '-1', '1',
     '-3', '-1', '-1', '-4'],
    ['-2', '-3', '-3', '-3', '-2', '-3', '-3', '-3', '-1', '0', '0', '-3', '0', '6', '-4', '-2', '-2', '1', '3', '-1',
     '-3', '-3', '-1', '-4'],
    ['-1', '-2', '-2', '-1', '-3', '-1', '-1', '-2', '-2', '-3', '-3', '-1', '-2', '-4', '7', '-1', '-1', '-4', '-3',
     '-2', '-2', '-1', '-2', '-4'],
    ['1', '-1', '1', '0', '-1', '0', '0', '0', '-1', '-2', '-2', '0', '-1', '-2', '-1', '4', '1', '-3', '-2', '-2', '0',
     '0', '0', '-4'],
    ['0', '-1', '0', '-1', '-1', '-1', '-1', '-2', '-2', '-1', '-1', '-1', '-1', '-2', '-1', '1', '5', '-2', '-2', '0',
     '-1', '-1', '0', '-4'],
    ['-3', '-3', '-4', '-4', '-2', '-2', '-3', '-2', '-2', '-3', '-2', '-3', '-1', '1', '-4', '-3', '-2', '11', '2',
     '-3', '-4', '-3', '-2', '-4'],
    ['-2', '-2', '-2', '-3', '-2', '-1', '-2', '-3', '2', '-1', '-1', '-2', '-1', '3', '-3', '-2', '-2', '2', '7', '-1',
     '-3', '-2', '-1', '-4'],
    ['0', '-3', '-3', '-3', '-1', '-2', '-2', '-3', '-3', '3', '1', '-2', '1', '-1', '-2', '-2', '0', '-3', '-1', '4',
     '-3', '-2', '-1', '-4'],
    ['-2', '-1', '3', '4', '-3', '0', '1', '-1', '0', '-3', '-4', '0', '-3', '-3', '-2', '0', '-1', '-4', '-3', '-3',
     '4', '1', '-1', '-4'],
    ['-1', '0', '0', '1', '-3', '3', '4', '-2', '0', '-3', '-3', '1', '-1', '-3', '-1', '0', '-1', '-3', '-2', '-2',
     '1', '4', '-1', '-4'],
    ['0', '-1', '-1', '-1', '-2', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-2', '0', '0', '-2', '-1',
     '-1', '-1', '-1', '-1', '-4'],
    ['-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4', '-4',
     '-4', '-4', '-4', '-4', '1']]


def get_score_62(item_1, item_2):
    score_62 = int(BLOSUM62_value[BLOSUM62_name[item_1]][BLOSUM62_name[item_2]])
    return score_62


def global_alignment(sequence_1, sequence_2, indel_penalty=-5):
    m = len(sequence_1)
    n = len(sequence_2)
    # get matrixes
    match_score_matrix = np.zeros((m + 1, n + 1))
    indel_score_1_matrix = np.zeros((m + 1, n + 1))
    indel_score_2_matrix = np.zeros((m + 1, n + 1))
    final_score_matrix = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        match_score_matrix[i][0] = indel_penalty * i
        indel_score_1_matrix[i][0] = indel_penalty * i
        indel_score_2_matrix[i][0] = indel_penalty * i
        final_score_matrix[i][0] = indel_penalty * i
    for j in range(n + 1):
        match_score_matrix[0][j] = indel_penalty * j
        indel_score_1_matrix[0][j] = indel_penalty * j
        indel_score_2_matrix[0][j] = indel_penalty * j
        final_score_matrix[0][j] = indel_penalty * j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = final_score_matrix[i - 1][j - 1] + get_score_62(sequence_1[i - 1], sequence_2[j - 1])
            indel_score_1 = final_score_matrix[i - 1][j] + indel_penalty
            indel_score_2 = final_score_matrix[i][j - 1] + indel_penalty
            final_score = max(match_score, indel_score_1, indel_score_2)
            match_score_matrix[i][j] = match_score
            indel_score_1_matrix[i][j] = indel_score_1
            indel_score_2_matrix[i][j] = indel_score_2
            final_score_matrix[i][j] = final_score
    # start searching
    alignment_1 = ''
    alignment_2 = ''
    i = m
    j = n
    score_sum = 0
    while i > 0 or j > 0:
        if i > 0 and final_score_matrix[i][j] == indel_score_1_matrix[i][j]:
            alignment_1 = alignment_1 + sequence_1[i - 1]
            alignment_2 = alignment_2 + '-'
            i = i - 1
            score_sum = score_sum + indel_penalty
        elif j > 0 and final_score_matrix[i][j] == indel_score_2_matrix[i][j]:
            alignment_1 = alignment_1 + '-'
            alignment_2 = alignment_2 + sequence_2[j - 1]
            j = j - 1
            score_sum = score_sum + indel_penalty
        elif i > 0 and j > 0 and final_score_matrix[i][j] == match_score_matrix[i][j]:
            alignment_1 = alignment_1 + sequence_1[i - 1]
            alignment_2 = alignment_2 + sequence_2[j - 1]
            i = i - 1
            j = j - 1
            score_sum = score_sum + get_score_62(sequence_1[i], sequence_2[j])
    alignment_1 = alignment_1[::-1]
    alignment_2 = alignment_2[::-1]
    return alignment_1, alignment_2, score_sum


if __name__ == '__main__':
    # # for degugging
    # debug_input_data_folder = './data/hw3q1/1_global_alignment_with_a_fixed_indel_penalty/1_global_alignment_with_a_fixed_indel_penalty/Debugging/inputs/'
    # debug_result_data_folder = './data/hw3q1/1_global_alignment_with_a_fixed_indel_penalty/1_global_alignment_with_a_fixed_indel_penalty/Debugging/outputs/'
    # for i in range(3):
    #     input_file_name = debug_input_data_folder + 'input_' + str(i + 1) + '.txt'
    #     with open(input_file_name, 'r') as input_file:
    #         data = input_file.read()
    #     sequence_1 = data.split('\n')[1]
    #     sequence_2 = data.split('\n')[3]
    #     result_file_name = debug_result_data_folder + 'output_' + str(i + 1) + '.txt'
    #     with open(result_file_name, 'r') as output_file:
    #         data = output_file.read()
    #     score_true = int(data.split('\n')[0])
    #     alignment_1_true = data.split('\n')[1]
    #     alignment_2_true = data.split('\n')[2]
    #     alignment_1, alignment_2, score = global_alignment(sequence_1, sequence_2, indel_penalty=-5)
    #     print('======================================================')
    #     print('round: ', str(i + 1))
    #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    #     print('result: ')
    #     print('score: ', score)
    #     print('alignment 1: ', alignment_1)
    #     print('alignment 2: ', alignment_2)
    #     print('len alignment 1: ', len(alignment_1))
    #     print('len alignment 2: ', len(alignment_2))
    #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    #     print('true: ')
    #     print('score true: ', score_true)
    #     print('alignment 1 true: ', alignment_1_true)
    #     print('alignment 2 true: ', alignment_2_true)
    #     print('len alignment 1 true: ', len(alignment_1_true))
    #     print('len alignment 2 true: ', len(alignment_2_true))
    #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    #     print('compare: ')
    #     if alignment_1 == alignment_1_true:
    #         print('perfect 1')
    #     if alignment_2 == alignment_2_true:
    #         print('perfect 2')
    #     print('======================================================')
    # for test
    input_file_name = './test_1.txt'
    with open(input_file_name, 'r') as input_file:
        data = input_file.read()
        sequence_1 = data.split('\n')[1]
        sequence_2 = data.split('\n')[3]
    alignment_1, alignment_2, score = global_alignment(sequence_1, sequence_2, indel_penalty=-5)
    output_file = './output_q1_Xinyuan_Wang.txt'
    with open(output_file, 'a') as of:
        of.write(str(score) + '\n')
        of.write(alignment_1 + '\n')
        of.write(alignment_2 + '\n')
