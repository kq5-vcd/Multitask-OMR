import cv2

def normalize(image):
    return (255. - image)/255.


def resize(image, height):
    #new_width = old_width * (new_height/old_height)
    width = int(float(height * image.shape[1]) / image.shape[0]) 
    sample_img = cv2.resize(image, (width, height))
    
    del width
    
    return sample_img


def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]/m


def sequence_levenshtein(a,b):
    n, m = len(a), len(b)
    dist = 0

    if n > m:
        a,b = b,a
        n,m = m,n
        
    for i in range(n):
        word = a[i]
        
        if len(b) == 1:
            dist += levenshtein(word, b[0])
            break
            
        b_seq = b[:-n+i+1]
        
        seq_dist = [levenshtein(word, x) for x in b_seq]
        best_val = min(seq_dist)
        dest_idx = seq_dist.index(best_val)
        
        dist += dest_idx + best_val
        
        if i == n - 1:
            dist += len(b) - dest_idx - 1
            break
            
        b = b[dest_idx+1:]
            

    return dist/m


def load_symbols(dict_path):
    word2int = {} #map symbols to numeric values
    int2word = {} #map numeric values to symbols

    dict_file = open(dict_path,'r')
    dict_list = dict_file.read().splitlines()
    
    for word in dict_list:
        if not word in word2int:
            word_idx = len(word2int)
            word2int[word] = word_idx
            int2word[word_idx] = word

    dict_file.close()
    
    del dict_file
    
    return (word2int, int2word, len(int2word))


def log_softmax_to_string(arr, int2word, blank):
    result = []
    for i in range(len(arr)):
        if arr[i] != blank:
            if i == 0 or arr[i] != arr[i-1]:
                result.append(int2word[arr[i]])
            
    return result