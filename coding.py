def solution(cards) :
    answer = []
    numbers = []
    letters = []

    for card in cards : 
        numbers.append(card[0])
        letters.append(card[1])

    if  numbers[1] > numbers[0] and numbers[1] == numbers[2]:
        answer.append("lose")

    return answer

answer = solution(["1r", "9b", "9r"])

# print(num)
# print(let)
print(answer)