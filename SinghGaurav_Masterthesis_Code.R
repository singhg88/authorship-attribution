################ Part 1: Classification ###################

setwd("C:/Users/Gaurav Singh/Desktop/Code/Corpus2/")
library(stylo)

raw.corpus <- load.corpus(files = "all")
summary(raw.corpus)

# Tokenize the corpus
tokenized.corpus <- txt.to.words.ext(raw.corpus,
                                     language= "English.all", 
                                     preserve.case=FALSE)

# Select the feature: Here the 200 Most Frequent Words  (MFT)
myFrequentFeatures <- make.frequency.list(tokenized.corpus,head=200)

# Represent each text with these 200 MFW
freqs <- make.table.of.frequencies(tokenized.corpus,
                                   features = myFrequentFeatures)

# Split the test and the training labels
test.labels <- c("Test_49.txt","Test_50.txt",
                 "Test_51.txt","Test_52.txt",
                 "Test_53.txt","Test_54.txt",
                 "Test_55.txt","Test_56.txt",
                 "Test_57.txt","Test_58.txt",
                 "Test_62.txt","Test_63.txt")

# Represent each text in the test set with these 200 MFW
freqs.test <- freqs[(rownames(freqs) %in% test.labels),]

# Represent each text in the training set with these 200 MFW
freqs.train <- freqs[!(rownames(freqs) %in% test.labels),]

# Apply the Delta method with 200 WFW
delta=perform.delta(freqs.train,freqs.test,distance = "delta",
                    no.of.candidates = 3,z.scores.both.sets = TRUE)

# SVM method
svm=perform.svm(freqs.train,freqs.test,no.of.candidates = 3,
                svm.kernel="linear",svm.degree = 3,svm.coef0 = 0,
                svm.cost = 1)

# Naive Bayes method
naiveBayes=perform.naivebayes(freqs.train,freqs.test,
                              classes.training.set = 3,
                              classes.test.set = 3)

# Nearest Shrunken Centroid method
nsc=perform.nsc(freqs.train,freqs.test,show.features = FALSE,
                no.of.candidates = 3)


################ Part 2: Cross validation ###################
setwd("C:/Users/Gaurav Singh/Desktop/Code/Corpus5/")
library(stylo)
raw.corpus <- load.corpus(files = "all")
summary(raw.corpus)

tokenized.corpus1 <- txt.to.words.ext(raw.corpus,
                                     language= "English.all", 
                                     preserve.case=FALSE)

myFrequentFeatures <- make.frequency.list(tokenized.corpus1,head=500)

# Represent each text with these 200 MFW
freqs <- make.table.of.frequencies(tokenized.corpus1,
                                   features = myFrequentFeatures)


set.seed(1257)
n <- dim(freqs)[1]
mySample <- 1:n
test <- sample(mySample, n*0.3) 
train <- mySample[-test]
text.train=freqs[train,]
text.test=freqs[test,]


results_delta <- classify(training.frequencies = text.train,
                    test.frequencies = text.test,
                    mfw.min = 50, mfw.max = 500, mfw.incr = 50,
                    classification.method = "delta", cv.folds = 10, 
                    gui = FALSE)

results_delta$cross.validation.summary
results_delta$features.actually.used
results_delta$overall.success.rate


results_svm <- classify(training.frequencies = text.train,
                          test.frequencies = text.test,
                          mfw.min = 50, mfw.max = 500, mfw.incr = 50,
                          classification.method = "svm", cv.folds = 10, 
                        gui = FALSE)

results_svm$cross.validation.summary
results_svm$overall.success.rate


results_naiveBayes <- classify(training.frequencies = text.train,
                          test.frequencies = text.test,
                          mfw.min = 50, mfw.max = 500, mfw.incr = 50,
                          classification.method = "naiveBayes", 
                          cv.folds = 10,gui = FALSE)

results_naiveBayes$cross.validation.summary
results_naiveBayes$overall.success.rate


results_nsc <- classify(training.frequencies = text.train,
                          test.frequencies = text.test,
                          mfw.min = 50, mfw.max = 500, mfw.incr = 50,
                          classification.method = "nsc", cv.folds = 10, 
                        gui = FALSE)

results_nsc$cross.validation.summary
results_nsc$overall.success.rate

colMeans(results_nsc$cross.validation.summary)
colMeans(results_naiveBayes$cross.validation.summary)
colMeans(results_svm$cross.validation.summary)
colMeans(results_delta$cross.validation.summary)


################ Part 3: Robustness check (tests) ###################

#mcnemar.test(results_svm$cross.validation.summary,
 #            results_nsc$cross.validation.summary)

wilcox.test(x=results_nsc$cross.validation.summary,
            y=results_svm$cross.validation.summary,
            alternative="greater")

