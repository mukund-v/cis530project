def find_answer(context, answer):
    ind = -1
    try: 
        context.find(answer)
    except as e:
        print ("Error", e)
    return ind
    