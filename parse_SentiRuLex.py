def sentiRuLex():
    dict = {}
    with open('SentiRuLex.txt', 'r') as f:
        for line in f:
            lex = line.split(',')
            # print(lex)
            # print(len(lex))
            # print('=')
            if lex[3] == ' negative' or lex[3] == ' positive':
                dict[lex[2][1:]] = (lex[1][1:], lex[3][1:], lex[4][1:])

    print(len(dict.keys()))
    return dict


if __name__ == '__main__':
    l = sentiRuLex()

    for i in l.keys():
        print(l[i])

