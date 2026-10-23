def encode(message):
    vowels = 'aeiouAEIOU'
    result = []
    for char in message:
        if char in vowels:
            # Replace vowel with letter 2 places ahead, preserving case
            if char.islower():
                new_char = chr(ord(char) + 2)
            else:
                new_char = chr(ord(char) + 2)
            # Swap case of the new character
            result.append(new_char.swapcase())
        else:
            result.append(char.swapcase())
    return ''.join(result)


def check(candidate):

    # Check some simple cases
    assert candidate('TEST') == 'tgst', "This prints if this assert fails 1 (good for debugging!)"
    assert candidate('Mudasir') == 'mWDCSKR', "This prints if this assert fails 2 (good for debugging!)"
    assert candidate('YES') == 'ygs', "This prints if this assert fails 3 (good for debugging!)"
    
    # Check some edge cases that are easy to work out by hand.
    assert candidate('This is a message') == 'tHKS KS C MGSSCGG', "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate("I DoNt KnOw WhAt tO WrItE") == 'k dQnT kNqW wHcT Tq wRkTg', "This prints if this assert fails 2 (also good for debugging!)"



check(encode)
