import random

def train(s):
    """ Trains the model by giving it a dictionary of the transition probabilities. """
    # Creates a new dictionary.
    newDict = {}
    # Separates the string into a list based on the spaces between words.
    wordList = s.split(' ')
    # Loops through the entire list of words. If the word is not in the dictionary, add the current word as a key and add the next word to its list. If the word is in the dictionary, add the next word to the list of the current word. Accounts for the last word in the list by performing the same operation on the first word in the list so that it connects.
    for num in range(len(wordList)):
        if num == len(wordList) - 1:
            if wordList[num] not in newDict:
                newDict[wordList[num]] = []
                newDict[wordList[num]].append(wordList[0])
            else:
                newDict[wordList[num]].append(wordList[0])
        else:
            if wordList[num] not in newDict:
                newDict[wordList[num]] = []
                newDict[wordList[num]].append(wordList[num + 1])
            else:
                newDict[wordList[num]].append(wordList[num + 1])

    return newDict

#string = "Yeah baby I like it like that You gotta believe me when I tell you I said I like it like that"

#string2 = "Hello, I am named Stephen. I am a very cool guy. My friend is named Peter."
#print(train(string2))


# When this function is used on a variable with a set of words, model is the 
# variable we want to access the words from, firstWord is the first word 
# it will start generating the words from, and the numWords is how many words
# down the list we want to generate.

def generate(model, firstWord, numWords):
    # Creates a new string and list with firstWord inside.
    newString = ""
    newString += firstWord
    wordList = []
    wordList.append(firstWord)
    # Loops through the number of words and randomly chooses the next word for each word that is being added to the list. Once the loop is done, the full string is returned.
    for x in range(numWords):
        nextWord = random.choice(model[wordList[x]])
        newString += " " + nextWord
        wordList.append(nextWord)
    return newString

#cardiB = train("Yeah baby I like it like that You gotta believe me when I tell you I said I like it like that")
#print(generate(cardiB, "I", 10))
