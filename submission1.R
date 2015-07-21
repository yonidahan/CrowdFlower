



setwd("C:/Users/Sarah/Desktop/Data Science/Projects/crowdflower")


#Libraries
lapply(c("irlba","tm","caret","Metrics","e1071","rARPACK","Matrix","dplyr","stringdist","XML"
         ,"kernlab","readr","slam","doParallel","foreach","lsa","stringr"),
       require,character.only=T)

if (Sys.getenv("JAVA_HOME")!="")#Java path
        Sys.setenv(JAVA_HOME="")
library("RWeka")




#Loading
train<-read_csv("train.csv")
test<-read_csv("test.csv")

#Outliers
train$row<-1:dim(train)[1]
out1<-train$row[train$relevance_variance==1.47]
out2<-train$row[train$relevance_variance==1.414]
out3<-train$row[train$relevance_variance==1.374]
out4<-train$row[train$relevance_variance==1.356]

train<-train[-c(out1,out2),]

y<-train$median_relevance


data<-select(train,query,product_title,product_description)
data<-rbind(data,select(test,query,product_title,product_description))




#Feature Engineering

#From html to text
html_to_text<-function(x){
        x<-htmlParse(x,asText=T)
        x<-xpathSApply(x,"//p",xmlValue)
        x<-paste0(x,collapse="\n")
        x<-str_replace_all(x,"[\r\n]"," ")        
}


#Combine title and description, remove html 
combine<-function(x){
        results<-Corpus(DataframeSource(x))
        results<-tm_map(results,content_transformer(function(x)iconv(enc2utf8(x),sub="byte")))
        results<-tm_map(results,content_transformer(html_to_text))
        results<-data.frame(text=unlist(sapply(results, `[`, "content")),stringsAsFactors=F)[,1]
        
        return(results)
}


data$results<-combine(select(data,product_title,product_description))



#Number of words
n_words<-function(string){
        return(sapply(gregexpr("[[:alpha:]]+", string), function(x) sum(x > 0)))}

data$qwords<-apply(data[1],1,n_words)
data$ptwords<-apply(data[2],1,n_words)
data$reswords<-apply(data[3],1,n_words)



#Cleaning
trans<-function(x){
        x<-VCorpus(VectorSource(x))
        x<-tm_map(x,content_transformer(function(x)iconv(enc2utf8(x),sub="byte")))
        x<-tm_map(x,removePunctuation)
        x<-tm_map(x,content_transformer(tolower))
        x<-tm_map(x,removeWords,stopwords("english"))
        x<-tm_map(x,stemDocument)
        x<-tm_map(x,stripWhitespace)
        x<-unlist(sapply(x,'[', "content"))
        attr(x,"names")<-NULL
        return(x)
}

data[,c(1,2,4)]<-apply(data[,c(1,2,4)],2,trans)
data<-as.data.frame(data,stringsAsFactors=F)



#Metrics

#Query-Results
data$cos_res<-stringdist(data$query,data$results,"cosine")
data$jac_res<-stringdist(data$query,data$results,"jaccard")
data$jw_res<-stringdist(data$query,data$results,"jw")
data$osa_res<-stringdist(data$query,data$results,"osa")
data$lcs_res<-stringdist(data$query,data$results,"lcs")


#Query-Title
data$cos_pt<-stringdist(data$query,data$product_title,"cosine")
data$jac_pt<-stringdist(data$query,data$product_title,"jaccard")
data$jw_pt<-stringdist(data$query,data$product_title,"jw")
data$osa_pt<-stringdist(data$query,data$product_title,"osa")
data$lcs_pt<-stringdist(data$query,data$product_title,"lcs")



#Mean of variances by specific query
train$query<-trans(train$query)

data$query<-as.factor(data$query)
train$relevance_variance<-as.numeric(train$relevance_variance)
data$row<-c(1:dim(data)[1])#In order to preserve the original order after merging

sum<-summarise(group_by(train,query),mean(relevance_variance))
names(sum)<-c("query","mean_relevance")

data<-merge(data,sum,by="query")
data<-data[order(data$row,decreasing=F),]
rm(sum)



#Mean of median_relevance...

sum<-summarise(group_by(train,query),mean(median_relevance))
names(sum)<-c("query","mean_median")

data<-merge(data,sum,by="query")
data<-data[order(data$row,decreasing=F),]
rm(sum)


#Number of matching words Query-Results
matching<-function(x){
        
        terms1<-x[1]
        terms2<-x[2]
        
        corpus<-Corpus(VectorSource(list(terms1,terms2)))
        corpus<-tm_map(corpus,content_transformer(function(x)iconv(enc2utf8(x),sub="byte")))
        corpus<-tm_map(corpus,content_transformer(tolower))
        corpus<-tm_map(corpus,removePunctuation)
        corpus<-tm_map(corpus,removeWords,stopwords("english"))
        corpus<-tm_map(corpus,stemDocument)
        corpus<-tm_map(corpus,stripWhitespace)
        freq<-DocumentTermMatrix(corpus)
        freq<-as.matrix(freq)
        matching<-sum(freq[1,]>0&freq[2,]>0)
        
        return(matching)               
}
data$matching<-apply(data[,c(1,3)],1,matching)


#Select additional features
metr<-select(data,jac_res,jw_res,cos_pt,lcs_pt,mean_relevance,mean_median,
             matching,qwords,reswords)


#Document-Term Matrix

txt<-paste(data$query,data$product_title)

dtm<-VCorpus(VectorSource(txt))

control<-list(weighting=function(x)weightTfIdf(x,normalize=T),
              tokenize=function(x)NGramTokenizer(x,Weka_control(min=1,max=3)),
              wordLength=c(1,Inf))

dtm<-DocumentTermMatrix(dtm,control=control)

m<-dtm

m<-removeSparseTerms(m,0.9993)

m<-Matrix(as.matrix(m),sparse=T)



#Split to train and test
train.data<-m[1:nrow(train),]
test.data<-m[-c(1:nrow(train)),]

train.metr<-as.matrix(metr[1:nrow(train),])
test.metr<-as.matrix(metr[-c(1:nrow(train)),])



#Parallel backend
nfolds<-5
ncores=4
cl<-makeCluster(ncores)
registerDoParallel(cl)



#Cross-Validation
tune.grid = expand.grid(ncomp=c(280,300),cost=c(80,100));

# run grid search in parallel
results = 
        foreach(gridId = c(1:nrow(tune.grid)), .packages=c('kernlab','rARPACK','caret','Metrics','Matrix'),
                .combine=rbind, .multicombine=T) %dopar% {
                        set.seed(2603)
                        
                
                        folds = createFolds(as.factor(y),nfolds);
                        svm_cost = tune.grid[gridId,"cost"];
                        svd_ncomp = tune.grid[gridId,"ncomp"];
                        q.kappa = 0;
                        
                        
                        for(i in 1: length(folds)){
                                
                                
                                smpl<-folds[[i]];
                                g_train<-train.data[-smpl,];
                                g_test<-train.data[smpl,];
                                y_train<-y[-smpl];
                                y_test<-y[smpl];
                                train_metr_cv<-train.metr[-smpl,]
                                test_metr_cv<-train.metr[smpl,]
                                
                                #svd 
                                g_train_svd<-svds(g_train,k=svd_ncomp,nv=svd_ncomp,nu=0)
                                
                                
                                #Apply mapping
                                g_train<-g_train%*%g_train_svd$v
                                g_test<-g_test%*%g_train_svd$v
                                
                                
                                #Include distance metrics
                                
                                g_train<-cbind(g_train,train_metr_cv)
                                g_test<-cbind(g_test,test_metr_cv)
                                
                                
                                #Classifier : svm
                                sv<-ksvm(as.matrix(g_train),as.factor(y_train),kernel="laplacedot",
                                         weights=c("1"=w_train[1],
                                                   "2"=w_train[2],
                                                   "3"=w_train[3],
                                                   "4"=w_train[4]),
                                          C=svm_cost,scaled=T,type="C-bsvc")
                                
                                
                                p<-predict(sv,newdata=g_test,type="response");
                                
                                
                                # calc the quadratic kappa
                                q.kappa<-q.kappa + ScoreQuadraticWeightedKappa(as.numeric(as.character(y_test)),
                                                                                as.numeric(as.character(p)));
                        }
                        return (c(
                                "qkapp"=q.kappa/length(folds),
                                "svd_ncomp"=svd_ncomp,
                                "svm_cost"=svm_cost
                        ));
                }
stopCluster(cl);

# get best results
results = data.frame(results,row.names=NULL);
print(results);


best.result<-results[order(results$qkapp,decreasing=T),][1,]
print(best.result)


# svd 
g_train_svd = svds(train.data,k = best.result$svd_ncomp,nv =best.result$svd_ncomp, nu=0);

train.data.fin = train.data %*% g_train_svd$v;
train.data.fin<-cbind(train.data.fin,train.metr)

test.data.fin = test.data %*% g_train_svd$v;
test.data.fin<-cbind(test.data.fin,test.metr)


sv_model = ksvm(as.matrix(train.data.fin),as.factor(y),
                kernel="laplacedot",
                C=best.result$svm_cost,
                scaled=T,
                weights=c("1"=w_train[1],
                          "2"=w_train[2],
                          "3"=w_train[3],
                          "4"=w_train[4]),
                type="C-bsvc")

sub.p = predict(sv_model,newdata=test.data.fin,type="response");

table(sub.p);

write.csv(data.frame("id"=test$id, "prediction"=sub.p),"submissionlast1.csv",quote=F,row.names=F)



