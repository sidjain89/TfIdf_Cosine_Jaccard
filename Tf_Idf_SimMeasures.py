import numpy as np
import os

# global variables
termsDocsArray = [] # array of arrays for terms by document
allTerms = [] # to hold all terms
tfidfDocsVector = []
folder = ""  # folder containing the tokenized / stemmed files
#end vars


###############################################################
# function to calculate tf                                    #
# @param termsInDoc : total no of words in current document   #  
# @param termToCheck : word for which we need to calculate TF #
# @return tf(term frequency) of term termToCheck              #
###############################################################
def TFCalculator(termsInDoc, termToCheck):
    ctr = 0
    for term in termsInDoc:
        if(str(term).lower() == termToCheck):
            ctr += 1
    return float(ctr / len(termsInDoc))
#end def

##############################################################
# Calculates idf of term termToCheck                         #
# @param allTerms : all the terms of all the documents       #
# @param termToCheck : need IDF for this word                #
# @return idf(inverse document frequency) score              #
##############################################################
def IDFCalculator(allTerms, termToCheck):
    ctr = 0
    for term in allTerms:
        if(str(term).lower() == termToCheck):
            ctr += 1
    return (np.log10(len(allTerms) / ctr) + 1)
# end def

###############################################
# Parses the files and adds terms into arrays #
# Fill allTerms and termsDocsArray            #
###############################################
def ParseFiles():
    for file in os.listdir(folder):
        name = str(file)
        fileFilter = "_token_stem" # filter criteria to select files - optional
        val = name.find(fileFilter)
        if(val <> -1):
            print("generating tf idf vector from file: " + str(file))
            fo = open(folder + str(file))
            text = fo.read()

            global allTerms
            global termsDocsArray
             
            allTerms.extend(text.split(" "))          
            termsDocsArray.append(text.split(" "))           
            fo.close()
        #end if
    #end for
    allTerms = np.unique(allTerms)
# end def

###############################################
# Main method to create TF IDF vector         #
###############################################
def TFIDFCalculator():
    tf = 0.0
    idf = 0.0
    tfidf = 0.0

    for docTerms in termsDocsArray:
        tfidfVector = range(len(allTerms))
        ctr = 0
        for term in docTerms:
            tf = TFCalculator(termsDocsArray, term)
            idf = IDFCalculator(allTerms, term)
            tfidf = tf * idf
            tfidfVector[ctr] = tfidf
            ctr += 1
        # end for
        tfidfDocsVector.append(tfidfVector)
        tfidfVector = [] # flush logic
    #end for
#end def

##############################################
# method to calculate cosine similarity      #
# @param docVector1 : document vector 1 (a)  #
# @param docVector2 : document vector 2 (b)  #
##############################################
def CosineSimilarity(docVector1, docVector2):
    dotProduct = 0.0
    magnitude1 = 0.0
    magnitude2 = 0.0
    cosineSimilarity = 0.0

    for i in range(0, len(docVector1) - 1):
        dotProduct += docVector1[i] * docVector2[i]
        magnitude1 += np.power(docVector1[i], 2)
        magnitude2 += np.power(docVector2[i], 2)
    # end for
    magnitude1 = np.sqrt(magnitude1)
    magnitude2 = np.sqrt(magnitude2)

    if(magnitude1 != 0.0 and magnitude2 != 0.0):
        cosineSimilarity = dotProduct / (magnitude1 * magnitude2)

    return(str(cosineSimilarity))
# end def

###################################################
# method to calculate jaccard score(bit vectors)  #
# @param docVector1 : document vector 1 (a)       #
# @param docVector2 : document vector 2 (b)       #
# formula = v1 . v2 / (|v1|^2 + |v2|^2 - v1 . v2) #
###################################################
def JaccardSimilarity(docVector1, docVector2):
    dotProduct = 0.0
    magnitude1 = 0.0
    magnitude2 = 0.0
    jaccardSimilarity = 0.0

    for i in range(0, len(docVector1) - 1):
        dotProduct += docVector1[i] * docVector2[i]
        magnitude1 += np.power(docVector1[i], 2)
        magnitude2 += np.power(docVector2[i], 2)
    # end for

    if(magnitude1 != 0.0 and magnitude2 != 0.0):
        jaccardSimilarity = dotProduct / (magnitude1 + magnitude2 - dotProduct)

    return(str(jaccardSimilarity))
# end def


############ main logic to call all def ##########
ParseFiles()
TFIDFCalculator()
# print(tfidfDocsVector)
print(CosineSimilarity(tfidfDocsVector[0], tfidfDocsVector[1]))
#print(JaccardSimilarity(tfidfDocsVector[0], tfidfDocsVector[1]))

# another way to calculate jaccard index - without using bit vectors
set1 = termsDocsArray[0]
set2 = termsDocsArray[1]
union = set(set1).union(set(set2))
intersect = set(set1).intersection(set(set2))
#print(len(intersect))
#print(len(union))
#print(type(intersect))
jaccardScore = float(len(intersect)) / float(len(union))
print("Jaccard Score: " + str(jaccardScore))
print("Done!")
####################### end of script ####################################