library(tm)
library(readr)
library(ngram)
library(RTextTools)
library(RWeka)
library(e1071)
file.remove("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/new.csv")

file.remove("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/label.csv")
folder1 <- "E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/pos/"      
file_list1 <- list.files(path=folder1, pattern="*.txt")
folder2 <- "E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/neg/"     
file_list2 <- list.files(path=folder2, pattern="*.txt")

min<-1
max<-500
ratio<-0.7
test<-(2*ratio*max+1):(2*max)
train<-min:(2*ratio*max)
for( i in min:max)
{
  
  x<-paste(folder1,file_list1[i],sep="")
  val<-read_file(x)
  # temp<-paste(0,val,sep = ",")
  write.table(val,"E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/new.csv",append = T,row.names = F,col.names = F,sep=",")
  write.table(0,"E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/label.csv",append = T,row.names = F,col.names = F,sep=",")
  
  #  f<-read.csv("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/new.csv",header = F)
  file.copy(x,"E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/temp/")
  name<-paste(formatC(i, width=3, flag="0"),"_pos",".txt",sep="")
  file.rename(from = file.path("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/temp/",file_list1[i]),to=file.path("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/temp/",name))
  
  y<-paste(folder2,file_list2[i],sep="")
  val<-read_file(y)
  write.table(1,"E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/label.csv",append = T,row.names = F,col.names = F,sep=",")
  
  write.table(val,"E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/new.csv",append = T,row.names = F,col.names = F,sep=",")
  file.copy(y,"E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/temp/")
  name<-paste(formatC(i, width=3, flag="0"),"_neg",".txt",sep="")
  file.rename(from = file.path("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/temp/",file_list2[i]),to=file.path("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/temp/",name))
  
}


obj<-read.table("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/new.csv")
obj<-as.data.frame(obj)

tdm.generate <- function(ng){
  
  corpus = Corpus(DataframeSource(obj))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus<- tm_map(corpus,removePunctuation) 
  corpus<- tm_map(corpus,removeWords, stopwords("english"))
  corpus<- tm_map(corpus,removeNumbers) 
  corpus <- tm_map(corpus, stemDocument, language = "english") 
  
  # options(mc.cores=1) # http://stackoverflow.com/questions/17703553/bigrams-instead-of-single-words-in-termdocument-matrix-using-r-and-rweka/20251039#20251039
  BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ng, max = ng)) # create n-grams
  tdm<- DocumentTermMatrix(corpus,control = list(stopwords = TRUE,tokenize = BigramTokenizer,removePunctuation = T,removeNumbers = T,weighting = weightBin))
}
tdm <- tdm.generate(2)
tdm<-as.matrix(tdm)
#View(tdm)

file<-read.csv("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/label.csv",header = F)
container<-create_container(tdm,t(file),trainSize = train,testSize = test,virgin = F)
models <- train_models(container, algorithms=c("SVM"))
results <- classify_models(container, models)
analytics <- create_analytics(container, results)
summary(analytics)
acc_svm<-recall_accuracy(as.numeric(as.factor(file[test,])), results[ ,"SVM_LABEL"])
acc_svm

file<-read.csv("E:/Projects/Sentiment Analysis/Dataset/aclImdb/train/label.csv",header = F)
container<-create_container(tdm,t(file),trainSize = train,testSize = test,virgin = F)
models <- train_models(container, algorithms=c("MAXENT"))
results <- classify_models(container, models)
analytics <- create_analytics(container, results)
summary(analytics)
acc_max<-recall_accuracy(as.numeric(as.factor(file[test, ])), results[,"MAXENTROPY_LABEL"])
acc_max

model <- naiveBayes(x = tdm[train,], y =as.factor(unlist(file[train,])))
pre <- predict(model, tdm[test,])
length(pre)
file <- unlist(file)

mat<-table(pre,(file[test]))
accuracy <- sum(diag(mat)) / sum(mat)
precision <- diag(mat) / rowSums(mat)
recall <- (diag(mat) / colSums(mat))
accuracy
precision 
recall