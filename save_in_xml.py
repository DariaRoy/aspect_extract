def save_result(X_pure, y_result, name):

    result_file = open(name, 'w')
    id = 0
    prev = -1
    with open('SentiRuEval_rest_markup_test.xml', 'r') as f:
        for line in f:
            if '<aspects>' in line:
                print('\t\t<aspects1>',file=result_file)
                while True:
                    if len(X_pure) <= id or X_pure[id]['from'] < prev:
                        prev = -1
                        break

                    label = y_result[id]
                    # print id
                    if label:
                        X_pure[id]['word'] = X_pure[id]['word'].replace('"', '&quot;')
                        st = '\t\t\t<aspect category="Whole" from="'
                        st += str(X_pure[id]['from'])
                        st += '" mark="Rel" sentiment="neutral" term="'
                        st += str(X_pure[id]['word'])
                        st += '" to="'
                        st += str(X_pure[id]['to'])
                        st += '" type="explicit"/>'
                        print(st, file=result_file)
                        #print(('\t\t\t<aspect category="Whole" from="%s" mark="Rel" sentiment="neutral" term="%s" to="%s" type="explicit"/>' % (X_pure[id]['from'], X_pure[id]['word'], X_pure[id]['to'])).encode('utf-8'),file=result_file)

                    prev = X_pure[id]['to']
                    id += 1
            elif '</aspects>' in line:
                print ('\t\t</aspects1>',file=result_file)
            elif 'aspect' in line:
                pass
            else:
                print (line.rstrip(),file=result_file)

    result_file.close()

    print('Results have been writen at file ', name)

