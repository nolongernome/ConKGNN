import os


def load_dataset(path, size):
    dic = {}
    num = 0
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if num == 0:
                num = num + 1
                continue
            line = line.strip().split()
            # 术语集存在名词中的空格，保留1--4和最后一个，其余合并
            if len(line) > size and size == 6:
                # print(line)
                line[0] = line[0].strip('"')
                dic[line[0]] = [line[2].strip('"'), line[3].strip('"')]
                term = ""
                for i in range(4, len(line) - 1):
                    term = term + line[i]
                dic[line[0]].append(term.strip('"'))
                dic[line[0]].append(line[-1].strip('"'))
            elif len(line) == size:
                line[0] = line[0].strip('"')
                dic[line[0]] = []
                for i in range(2, len(line)):
                    dic[line[0]].append(line[i].strip('"'))

            else:
                print(line)
            num = num + 1
            if num == 100000000000:
                break
    print("加载数据条：" + str(num - 1))
    return dic  # [([...], 0), ([...], 1), ...]


def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False

