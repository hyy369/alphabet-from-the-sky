# !/usr/bin/python
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Set the directory you want to start from
    rootDir = '../data'
    char_count_dict = {}
    char_sum = 0

    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            fname_list = fname.split('_')
            character = fname_list[0]
            if character == '.DS':
                continue
            if character == 'colon': 
                character = ':'
            if character == 'question': 
                character = '?'
            if character == 'dot': 
                character = '.'
            readability = fname_list[1]
            char_sum += 1
            if character in char_count_dict:
                char_count_dict[character] += 1
            else:
                char_count_dict[character] = 1

    # Sort the dictionary by ascii order and plot bar graph
    sorted_keys = sorted(char_count_dict.keys())
    sorted_values = []
    for key in sorted_keys:
        sorted_values.append(char_count_dict[key])

    plt.bar(sorted_keys, sorted_values, color='b')
    plt.show()