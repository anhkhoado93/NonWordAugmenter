from typing import List, Union, Any

import numpy as np
import nlpaug.augmenter.char as nac
import numpy.random as random
from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Action, Method, Doc, LibraryUtil
import nlpaug.model.char as nmc
def defaultTokenizer(string, seperator=' '):
    tokenizedList = string.split(seperator)
    tokenizedList = list(filter(lambda token: token != '', tokenizedList))
    return tokenizedList


compositionChars = ['á', 'à', 'ạ', 'ả', 'ã', 'â', 'ấ', 'ầ', 'ậ', 'ẩ', 'ẫ', 'ă', 'ắ', 'ằ', 'ặ', 'ẳ',
                              'ẵ', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ',
                              'ự', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ó', 'ò', 'ỏ', 'õ',
                              'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'đ', 'ý', 'ỳ',
                              'ỷ', 'ỹ', 'ỵ']

####################### BASE CLASSES ###########################
class TypoAugmenter(nac.CharAugmenter):
    def __init__(self, name='TypoAugmenter', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.typo = {}
        self.eligibleCharacters = compositionChars

    def _isTypo(self, character):
        return character in self.typo

    def _randomDecomposeTwice(self):
        return np.random.uniform() < 0.50

    def _generateCDF(self, length):
        if length == 1: return [1]
        mid = length - 2
        end_prob = (1 - mid*0.1)/2
        return [end_prob] + [0.1]*mid + [end_prob]

    def _getNewDecomposition(self, baseWord, compWord):
        pass

    def _getTypoDictionary(self):
        pass

    def _containsUO(self, word):
        lstUO = ['ươ', 'ướ', 'ườ','ưở', 'ượ', "ưỡ"]
        return any(map(lambda x: x in word, lstUO))

    def _insertBaseWord(self,listOfChars, index, baseWord):
        if len(baseWord) == 1:
            listOfChars[index] = baseWord
        else:
            listOfChars[index] = baseWord[0]
            listOfChars[index+1] = baseWord[1]

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        aug_word_idxes = self._get_aug_idxes(tokens, \
                        self.aug_word_min, self.aug_word_max, \
                        self.aug_word_p, Method.WORD)
        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue
            result = self.generateWordError(token)
            results.append(result)
        return ' '.join(results)

    def generateWordError(self, word):
        # if not word[0].isalpha(): return word
        baseWord, comWord = "", []
        listOfChars = [w for w in word]
        index = -1
        if self._containsUO(word):
            index = listOfChars.index('ư')
            uoType = ''.join(listOfChars[index:index + 2])
            if uoType not in ['ươ', 'ướ', 'ườ', 'ưở', 'ượ', "ưỡ"]:
                return word
            baseWord, compWord = self.typo[uoType][0], self.typo[uoType][1:]
        else:
            index, telexChar = next(((i, c) for i, c in enumerate(listOfChars) \
                                     if self._isTypo(c)), (None,None))
            if not telexChar: return ''.join(listOfChars)
            baseWord, compWord = self.typo[telexChar][0], self.typo[telexChar][1:]

        isDecomposedTwice = len(compWord) != 1 and self._randomDecomposeTwice()
        if not isDecomposedTwice:
            baseWord, compWord = self._getNewDecomposition(baseWord, compWord)
        self._insertBaseWord(listOfChars, index, baseWord)
        self._insertRandom(listOfChars, index, baseWord, compWord[0])
        if isDecomposedTwice:
            self._insertRandom(listOfChars, index, baseWord, compWord[1])
        result = ''.join(listOfChars)

        return result

    def _insertRandom(self, listOfChars, indexOfBase, baseCharacter, telexCharacter):
        possibleIndices = list(range(indexOfBase +
                                   (1 if len(baseCharacter) == 2 else 0), len(listOfChars) + 1))
        cdf = self._generateCDF(len(possibleIndices))
        indexToInsert = np.random.choice(possibleIndices, p=cdf)
        if indexToInsert == len(listOfChars):
            listOfChars.append(telexCharacter)
        else:
            listOfChars[indexToInsert:indexToInsert] = telexCharacter


class AccentAugmenter(nac.CharAugmenter):
    def __init__(self, name='AccentAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.eligibleCharacters = compositionChars
        self.model = {}

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_min, \
                        self.aug_word_max, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            for char in chars:
                if char not in self.eligibleCharacters:
                    result += char
                    continue
                if random.random() < self.aug_char_p:
                    result += self.sample(self.model[char], 1)[0]
                else:
                    result += char

            results.append(result)

        return ' '.join(results)


###################################
class TelexAugmenter(TypoAugmenter):
    def __init__(self, name='TelexAugmenter', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.typo = self._getTypoDictionary()

    def _getTypoDictionary(self):
        return {
          "á": ["a","s"], "à": ["a","f"],
          "ạ": ["a","j"], "ả": ["a","r"],
          "ã": ["a","x"], "â": ["a","a"],
          "ấ": ["a","a","s"],"ầ": ["a","a","f"],
          "ậ": ["a","a","j"],"ẩ": ["a","a","r"],
          "ẫ": ["a","a","x"],"ă": ["a","w"],
          "ắ": ["a","w","s"],"ằ": ["a","w","f"],
          "ặ": ["a","w","j"],"ẳ": ["a","w","r"],
          "ẵ": ["a","w","x"],"í": ["i","s"],
          "ì": ["i","f"],"ỉ": ["i","r"],
          "ĩ": ["i","x"],"ị": ["i","j"],
          "ú": ["u","s"],"ù": ["u","f"],
          "ủ": ["u","r"],"ũ": ["u","x"],
          "ụ": ["u","j"],"ư": ["u","w"],
          "ứ": ["u","w","s"],"ừ": ["u","w","f"],
          "ử": ["u","w","r"],"ữ": ["u","w","x"],
          "ự": ["u","w","j"],"é": ["e","s"],
          "è": ["e","f"],"ẻ": ["e","r"],
          "ẽ": ["e","x"],"ẹ": ["e","j"],
          "ê": ["e","e"],"ế": ["e","e","s"],
          "ề": ["e","e","f"],"ể": ["e","e","r"],
          "ễ": ["e","e","x"],"ệ": ["e","e","j"],
          "ó": ["o","s"],"ò": ["o","f"],
          "ỏ": ["o","r"],"õ": ["o","x"],
          "ọ": ["o","j"],"ô": ["o","o"],
          "ố": ["o","o","s"],"ồ": ["o","o","f"],
          "ổ": ["o","o","r"],"ỗ": ["o","o","x"],
          "ộ": ["o","o","j"],"ơ": ["o","w"],
          "ớ": ["o","w","s"],"ờ": ["o","w","f"],
          "ở": ["o","w","r"],"ỡ": ["o","w","x"],
          "ợ": ["o","w","j"],'đ': ['d','d'],
          "ý": ["y","s"],"ỳ": ["y","f"],
          "ỷ": ["y","r"],"ỹ": ["y","x"],
          "ỵ": ["y","j"] , "ươ": ['uo', "w"],
          "ướ": ['uo', "w", "s"], "ườ": ['uo','w','f'],
            "ưỡ": ['uo', "w", "x"], "ưở": ['uo','w','r'],
            "ượ": ['uo', "w", "j"]}
    def _getNewDecomposition(self, baseWord,compWord):
        if len(compWord) == 1: return baseWord, compWord
        if baseWord == 'o':
            baseWord = ('ô' if compWord[0] == 'o' else 'ơ')
        elif baseWord == 'a':
            baseWord = ('â' if compWord[0] == 'a' else 'ă')
        elif baseWord == 'e': baseWord = 'ê'
        elif len(baseWord) == 2: baseWord = 'ươ'
        compWord = compWord[1:]
        return baseWord, compWord

class VNIAugmenter(TypoAugmenter):
    def __init__(self, name='VNIAugmenter', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.typo = self._getTypoDictionary()

    def _getTypoDictionary(self):
        return {
          "á": ["a","1"], "à": ["a","2"],
          "ạ": ["a","5"], "ả": ["a","3"],
          "ã": ["a","4"], "â": ["a","a"],
          "ấ": ["a","6","1"],"ầ": ["a","6","2"],
          "ậ": ["a","6","5"],"ẩ": ["a","6","3"],
          "ẫ": ["a","6","4"],"ă": ["a","7"],
          "ắ": ["a","7","1"],"ằ": ["a","7","2"],
          "ặ": ["a","7","5"],"ẳ": ["a","7","3"],
          "ẵ": ["a","7","4"],"í": ["i","1"],
          "ì": ["i","2"],"ỉ": ["i","3"],
          "ĩ": ["i","4"],"ị": ["i","5"],
          "ú": ["u","1"],"ù": ["u","2"],
          "ủ": ["u","3"],"ũ": ["u","4"],
          "ụ": ["u","5"],"ư": ["u","7"],
          "ứ": ["u","7","1"],"ừ": ["u","7","2"],
          "ử": ["u","7","3"],"ữ": ["u","7","4"],
          "ự": ["u","7","5"],"é": ["e","1"],
          "è": ["e","2"],"ẻ": ["e","3"],
          "ẽ": ["e","4"],"ẹ": ["e","5"],
          "ê": ["e","6"],"ế": ["e","6","1"],
          "ề": ["e","6","2"],"ể": ["e","6","3"],
          "ễ": ["e","6","4"],"ệ": ["e","6","5"],
          "ó": ["o","1"],"ò": ["o","2"],
          "ỏ": ["o","3"],"õ": ["o","4"],
          "ọ": ["o","5"],"ô": ["o","o"],
          "ố": ["o","6","1"],"ồ": ["o","6","2"],
          "ổ": ["o","6","3"],"ỗ": ["o","6","4"],
          "ộ": ["o","6","5"],"ơ": ["o","7"],
          "ớ": ["o","7","1"],"ờ": ["o","7","2"],
          "ở": ["o","7","3"],"ỡ": ["o","7","4"],
          "ợ": ["o","7","5"],'đ': ['d','9'],
          "ý": ["y","1"],"ỳ": ["y","2"],
          "ỷ": ["y","3"],"ỹ": ["y","4"],
          "ỵ": ["y","5"], 'ươ': ['uo', '7'],
            "ướ": ['uo', "7", "1"], "ườ": ['uo', '7', '2'],
            "ưỡ": ['uo', "7", "4"], "ưở": ['uo', '7', '3'],
            "ượ": ['uo', "7", "5"]
        }
    def _getNewDecomposition(self, baseWord,compWord):
        if len(compWord) == 1: return baseWord, compWord
        if baseWord == 'o':
            baseWord = ('ô' if compWord[0] == '6' else 'ơ')
        elif baseWord == 'a':
            baseWord = ('â' if compWord[0] == '6' else 'ă')
        elif baseWord == 'e': baseWord = 'ê'
        elif len(baseWord) == 2: baseWord = 'ươ'
        compWord = compWord[1:]
        return baseWord, compWord

###################################
class AccentAugmenter(nac.CharAugmenter):
    def __init__(self, name='AccentAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.eligibleCharacters = compositionChars
        self.model = {}
    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        print('tokens: ', tokens)
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)
        print(aug_word_idxes)
        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            for char in chars:
                if char not in self.eligibleCharacters:
                    result += char
                    continue
                if np.random.random() < self.aug_char_p:
                    result += self.sample(self.model[char], 1)[0]
                else:
                    result += char

            results.append(result)

        return ' '.join(results)

class MissingDialectAugmenter(AccentAugmenter):
    def __init__(self, name='MissingDialectAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)

        self.model = {
                "á": ["a"], "à": ["a"],
                "ạ": ["a"], "ả": ["a"],
                "ã": ["a"], "â": ["a"],
                "ấ": ["a", "â", "á"], "ầ": ["a", "â", "à"],
                "ậ": ["a", "â", "ạ"], "ẩ": ["a", "â", "ả"],
                "ẫ": ["a", "â", "ẫ"], "ă": ["a"],
                "ắ": ["a", "ă", "á"], "ằ": ["a", "ă", "à"],
                "ặ": ["a", "ă", "ạ"], "ẳ": ["a", "ă", "ạ"],
                "ẵ": ["a", "ă", "ã"], "í": ["i"],
                "ì": ["i"], "ỉ": ["i"],
                "ĩ": ["i"], "ị": ["i"],
                "ú": ["u"], "ù": ["u"],
                "ủ": ["u"], "ũ": ["u"],
                "ụ": ["u"], "ư": ["u"],
                "ứ": ["u", "ư", "ú"], "ừ": ["u", "ư", "ù"],
                "ử": ["u", "ư", "ủ"], "ữ": ["u", "ư", "ũ"],
                "ự": ["u", "ư", "ụ"], "é": ["e"],
                "è": ["e"], "ẻ": ["e"],
                "ẽ": ["e"], "ẹ": ["e"],
                "ê": ["e"], "ế": ["e", "ê", "é"],
                "ề": ["e", "ê", "è"], "ể": ["e", "ê", "ẻ"],
                "ễ": ["e", "ê", "ẽ"], "ệ": ["e", "ê", "ẹ"],
                "ó": ["o"], "ò": ["o"],
                "ỏ": ["o"], "õ": ["o"],
                "ọ": ["o"], "ô": ["o"],
                "ố": ["o", "ô", "ó"], "ồ": ["o", "ô", "ò"],
                "ổ": ["o", "ô", "ỏ"], "ỗ": ["o", "ô", "õ"],
                "ộ": ["o", "ô", "ọ"], "ơ": ["o"],
                "ớ": ["o", "ơ", "ó"], "ờ": ["o", "ơ", "ò"],
                "ở": ["o", "ơ", "ỏ"], "ỡ": ["o", "ơ", "õ"],
                "ợ": ["o", "ơ", "ọ"], 'đ': ['d'],
                "ý": ["y"], "ỳ": ["y"],
                "ỷ": ["y"], "ỹ": ["y"],
                "ỵ": ["y"]
            }
        self.eligibleCharacters = compositionChars

class NoDialectAugmenter(AccentAugmenter):
    def __init__(self, name='NoDialectAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)

        self.model = {
            "á": ["a"], "à": ["a"],
            "ạ": ["a"], "ả": ["a"],
            "ã": ["a"], "â": ["a"],
            "ấ": ["a"], "ầ": ["a"],
            "ậ": ["a"], "ẩ": ["a"],
            "ẫ": ["a"], "ă": ["a"],
            "ắ": ["a"], "ằ": ["a"],
            "ặ": ["a"], "ẳ": ["a"],
            "ẵ": ["a"], "í": ["i"],
            "ì": ["i"], "ỉ": ["i"],
            "ĩ": ["i"], "ị": ["i"],
            "ú": ["u"], "ù": ["u"],
            "ủ": ["u"], "ũ": ["u"],
            "ụ": ["u"], "ư": ["u"],
            "ứ": ["u"], "ừ": ["u"],
            "ử": ["u"], "ữ": ["u"],
            "ự": ["u"], "é": ["e"],
            "è": ["e"], "ẻ": ["e"],
            "ẽ": ["e"], "ẹ": ["e"],
            "ê": ["e"], "ế": ["e"],
            "ề": ["e"], "ể": ["e"],
            "ễ": ["e"], "ệ": ["e"],
            "ó": ["o"], "ò": ["o"],
            "ỏ": ["o"], "õ": ["o"],
            "ọ": ["o"], "ô": ["o"],
            "ố": ["o"], "ồ": ["o"],
            "ổ": ["o"], "ỗ": ["o"],
            "ộ": ["o"], "ơ": ["o"],
            "ớ": ["o"], "ờ": ["o"],
            "ở": ["o"], "ỡ": ["o"],
            "ợ": ["o"], 'đ': ['d'],
            "ý": ["y"], "ỳ": ["y"],
            "ỷ": ["y"], "ỹ": ["y"],
            "ỵ": ["y"]
        }
        self.eligibleCharacters = compositionChars

class WrongDialectAugmenter(AccentAugmenter):
    def __init__(self, name='WrongDialectAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)

        self.model = {
            "ả": ["ã"],
            "ã": ["ả"],
            "ẩ": ["ẫ"],
            "ẫ": ["ẩ"],
            "ẳ": ["ẵ"],
            "ẵ": ["ẳ"],
            "ỉ": ["ĩ"],
            "ĩ": ["ỉ"],
            "ủ": ["ũ"],
            "ũ": ["ủ"],
            "ử": ["ữ"],
            "ữ": ["ử"],
            "ẻ": ["ẽ"],
            "ẽ": ["ẻ"],
            "ể": ["ễ"],
            "ễ": ["ể"],
            "ỏ": ["õ"],
            "õ": ["ỏ"],
            "ổ": ["ỗ"],
            "ỗ": ["ổ"],
            "ở": ["ỡ"],
            "ỡ": ["ở"],
            "ỷ": ["ỹ"],
            "ỹ": ["ỷ"],
            "e": ["ê"],
            "o": ["ô", "ơ"],
            "u": ["ư"],
            "a": ["ă","â"]
        }
        self.eligibleCharacters = self.model.keys()


class MyKeyboardAug(nac.KeyboardAug):
    def __init__(self, name='MyKeyboardAug', aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_p=0.3, aug_word_min=1, aug_word_max=10, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, include_special_char=True, include_numeric=True,
                 include_upper_case=True, lang="en", verbose=0, stopwords_regex=None, model_path=None,
                 min_char=4):
        super().__init__(self, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                         aug_word_p=0.3, aug_word_min=1, aug_word_max=10, stopwords=None,
                         tokenizer=None, reverse_tokenizer=None, include_special_char=True, include_numeric=True,
                         include_upper_case=True, lang="en", verbose=0, stopwords_regex=None, model_path=None,
                         min_char=4)
        self.telexDecomposer = TelexAugmenter()

    # def skip_aug(self, token_idxes, tokens):
    #     results = []
    #     for token_idx in token_idxes:
    #         char = tokens[token_idx]
    #         if char in self.model.model and len(self.model.predict(char)) > 0:
    #             results.append(token_idx)
    #
    #     return results

    def substitute(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0

        doc = Doc(data, self.tokenizer(data))
        aug_word_idxes = self._get_aug_idxes(doc.get_original_tokens(), self.aug_word_min,
                                             self.aug_word_max, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(doc.get_original_tokens()):
            if token_i not in aug_word_idxes:
                continue

            new_token = ""
            token = self.telexDecomposer.generateWord(token)
            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max,
                                                 self.aug_char_p, Method.CHAR)

            if aug_char_idxes is None:
                continue

            for char_i, char in enumerate(chars):
                if char_i not in aug_char_idxes:
                    new_token += char
                    continue

                sampled = ""
                new_token += self.sample(self.model.predict(chars[char_i]), 1)[0]

            # No capitalization alignment as this augmenter try to simulate typo

            change_seq += 1
            doc.add_change_log(token_i, new_token=new_token, action=Action.SUBSTITUTE,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return ' '.join(doc.get_augmented_tokens()), doc.get_change_logs()
        else:
            return ' '.join(doc.get_augmented_tokens())

    @classmethod
    def get_model(cls, special_char=True, numeric=True, upper_case=False, lang="en", model_path=None):
        return nmc.Keyboard(special_char=special_char, numeric=numeric, upper_case=False, lang=lang,
                            model_path=model_path)

class MyRandomAugmenter(nac.CharAugmenter):
    def __init__(self, name='MyRandomAugmenter', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.telexDecomposer = TelexAugmenter()
        self.randomizer = [
          nac.RandomCharAug(action="insert",aug_word_p=1, min_char=3, include_upper_case=False),
          nac.RandomCharAug(action="delete",aug_word_p=1, min_char=3, include_upper_case=False),
          nac.RandomCharAug(action="substitute", aug_word_p=1, min_char=3, include_upper_case=False),
          nac.RandomCharAug(action="swap",aug_word_p=1, min_char=3, include_upper_case=False)
        ]
    def substitute(self, data):
        results = []
        # Tokenize a text (e.g. The quick brown fox jumps over the lazy dog) to tokens (e.g. ['The', 'quick', ...])
        tokens = self.tokenizer(data)
        # Get target tokens
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)
        for token_i, token in enumerate(tokens):
            # Do not augment if it is not the target
            if token_i not in aug_word_idxes:
                results.append(token)
                continue
            token = self.telexDecomposer.generateWord(token)
            action = np.random.choice(self.randomizer, p=[0.3, 0.3, 0.2, 0.2])
            result = action.augment(token)
            results.append(result)

        return ' '.join(results)

###Spelling Augmenter

finalConsonant = ['i','y','c','t','n','ng','nh']
beginConsonant = ['x','s','d','đ','c','k','ngh','ng','gh','g','gi','d','r','tr','ch','n','l','kh','qu','u','v','nh']
class SpellingReplacementAugmenter(nac.CharAugmenter):
    def __init__(self, name='SpellingReplacementAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.model_beginconsonant = {
            "x": ["s"],
            "s": ["x"],
            "d": ['đ'],
            "đ": ['d'],
            "c": ["k"],
            "k": ["c"],
            "ngh": ["ng"],
            "ng": ["ngh"],
            "gh": ["g"],
            "g": ["gh"],
            "gi": ["d","r","v"],
            "d": ["gi","r","v"],
            "v": ["gi","d"],
            "r": ["d","gi"],
            "tr": ["ch"],
            "ch": ["tr"],
            "n": ["l"],
            "l": ["n"],
            "kh": ["k"],
            "k": ["kh"],
            "qu" : ["u"],
            "u": ["qu"],
            "nh" : ["nh"]
        }
        self.model_finalconsonant ={
            "c": ["t"],
            "t": ["c"],
            "n": ["ng"],
            "ng": ["n"],
            "n": ["nh"],
            "nh": ["n"],
            "i": ["y"],
            "y": ["i"]
        }

    def check_pos_consonant(self, word, mode):
        if not any(char.isalpha() for char in word):
            return word
        if mode == 'begin':
            while not word[0].isalpha():
                word = word[1:]
            if word[0] in beginConsonant or word[:2] in beginConsonant or word[:3] in beginConsonant:
                return True
        else:
            while not word[-1].isalpha():
                word = word[:-1]
            if word[-1] in finalConsonant or word[-2:] in finalConsonant:
                return True
        return False

    def sample_uppercase(self, word, mode):
        result = ''
        if mode == 'begin':
            prefix = self.sample(self.model_beginconsonant[word.lower()], 1)[0]
            if word.isupper():
                result += prefix.upper()
            elif word[0].isupper():
                result += prefix[0].upper()
                if len(prefix)>1 : result += prefix[1:] 
            else:
                result += prefix
        else:
            prefix = self.sample(self.model_finalconsonant[word.lower()], 1)[0]
            if word.isupper():
                result += prefix.upper()
            else:
                result += prefix
        return result

    def substitute_data(self, data, mode):
        tokens = self.tokenizer(data)
        new_tokens = []
        index_new_tokens = []
        for token_i, token in enumerate(tokens): 
          if self.check_pos_consonant(token.lower(), mode): 
            new_tokens.append(token)
            index_new_tokens.append(token_i)
        if new_tokens:
            aug_word_idxes = self._get_aug_idxes(new_tokens, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)
            for token_i, token in enumerate(new_tokens):
                if token_i not in aug_word_idxes:
                    continue
                result = ''
                if mode == 'begin':
                    while not token[0].isalpha():
                        result += token[0]
                        token = token[1:]
                    if token[:3].lower() in beginConsonant:
                        if np.random.random() < self.aug_char_p:
                            result += self.sample_uppercase(token[:3], mode)
                            result += token[3:]
                    elif token[:2].lower() in beginConsonant :
                        if np.random.random() < self.aug_char_p:
                            result = self.sample_uppercase(token[:2], mode)
                            result += token[2:]
                    else:
                        if np.random.random() < self.aug_char_p: 
                            result = self.sample_uppercase(token[:1], mode)
                            result += token[1:]
                else:
                    end_consonants = ''
                    while not token[-1].isalpha():
                        end_consonants += token[-1]
                        token = token[:-1]
                    if token[-2:].lower() in finalConsonant:
                        if np.random.random() < self.aug_char_p:
                            result = token[:-2] + self.sample_uppercase(token[-2:], mode) + end_consonants
                    else:
                        if np.random.random() < self.aug_char_p: 
                            result = token[:-1] + self.sample_uppercase(token[-1], mode) + end_consonants

                if result:
                    tokens[index_new_tokens[token_i]]= result

            return ' '.join(tokens)
        return data


class SpellingReplacementAugmenterBegin(SpellingReplacementAugmenter):
    def __init__(self, name='SpellingReplacementAugmenterBegin', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)
    
    def substitute(self, data):
        return self.substitute_data(data, 'begin')

class SpellingReplacementAugmenterFinal(SpellingReplacementAugmenter):
    def __init__(self, name='SpellingReplacementAugmenterFinal', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)
    
    def substitute(self, data):
        return self.substitute_data(data, 'final')

class SubsituteAugmenter(nac.CharAugmenter):
    def __init__(self, name='SubsituteAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.eligibleCharacters = compositionChars
        self.vowels = ['a','â','ă','e','ê','o','ô','ơ','u','ư','y']
        self.consonants_1 = ['b', 'd', 'h', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'x', 'đ', 'g','k','c'] 
        self.consonants_2 = ['tr', 'th', 'ch','ph', 'nh', 'kh','gi','qu','ng','gh']

    def check_if_vowel(self, char):
        return char.lower() in ['a','â','ă','e','ê','o','ô','ơ','u','ư','y']

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_min, \
                        self.aug_word_max, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            i = 0
            while (i < len(chars)):
                if random.random() < self.aug_char_p:
                    if self.check_if_vowel(chars[i]):
                        sub_char = self.sample(self.vowels, 1)[0]
                        if chars[i].isupper(): result += sub_char.upper()
                        else: result += sub_char
                    else:
                        if ''.join(chars[i:i+2]).lower() in self.consonants_2:
                            sub_char = self.sample(self.consonants_2, 1)[0]
                            if chars[i].isupper(): result += sub_char[0].upper()
                            else: result += sub_char[0]
                            if chars[i+1].isupper(): result += sub_char[1].upper()
                            else: result += sub_char[1]
                            i +=1
                        elif chars[i].lower() in self.consonants_1:
                            sub_char = self.sample(self.consonants_1, 1)[0]
                            if chars[i].isupper(): result += sub_char.upper()
                            else: result += sub_char
                        else:
                            result += chars[i]
                else:
                    if ''.join(chars[i:i+2]).lower() in self.consonants_2:
                        result += ''.join(chars[i:i+2])
                        i+=1
                    else: result += chars[i]
                i+=1

            results.append(result)

        return ' '.join(results)




