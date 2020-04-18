def find_answer(context, answer):
    ind = -1
    try:
        ind = context.find(answer)
    except as e:
        return -1, -1
    return ind, ind + len(answer)
