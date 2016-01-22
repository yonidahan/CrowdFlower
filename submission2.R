

setwd("C:/Users/Sarah/Desktop/Data Science/Projects/crowdflower")
lapply(c("dplyr","tm","caret","Metrics","e1071","rARPACK","Matrix","kernlab",
         "readr","slam","doParallel","foreach","lsa"),require,
       character.only=TRUE)
if (Sys.getenv("JAVA_HOME")!="")
        Sys.setenv(JAVA_HOME="")
library("RWeka");


#Stopwords dictionary
dict<-unique(c(stopwords("english"),stopwords("SMART")))


#Cleaning Function
trans<-function(x){
        x<-tm_map(x,content_transformer(function(x)iconv(enc2utf8(x),sub="byte")))
        x<-tm_map(x,content_transformer(tolower))
        x<-tm_map(x,removePunctuation)
        x<-tm_map(x,removeWords,stopwords("english"))
        x<-tm_map(x,stemDocument)
        x<-tm_map(x,stripWhitespace)
}



tfidf = function(txt,smooth_idf=T,sublinear_tf=F,
                 normf=NULL,
                 min_df=1,
                 do.trace=T,
                 use_idf = T,
                 ngram_range=NULL){
        #Create a corpus
        corp = NULL;
        if(do.trace) print("Building corpus!");
        if (!is.na(match(class(txt),c("VCorpus","Corpus")))) corp = txt;
        if (!is.na(match(class(txt),c("VectorSource","SimpleSource","Source")))) {corp = Corpus(txt);}
        if (class(txt) == "character") {corp = Corpus(VectorSource(txt));}
        if (is.null(corp)) {stop(paste("Error, unable to create a corpus", class(txt)));}
        
        #Corpora cleaning
        corp<-trans(corp)
        
        
        
        #document term matrix
        if(do.trace) print("Document-term matrix!");
        
        #ngram 
        Tokenizer = NULL;
        if (!is.null(ngram_range)){
                if(do.trace) print(paste("NGramTokenizer, range:",ngram_range[1],":",ngram_range[2]));
                Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ngram_range[1], max = ngram_range[2]))
                options(mc.cores=1)
                dtm = DocumentTermMatrix(corp,control=list(tokenize=Tokenizer,removePunctuation=T,wordLengths=c(1,10000),weighting=function(x) weightTf(x)))
        } else {
                dtm = DocumentTermMatrix(corp,control=list(removePunctuation=T,wordLengths=c(1,10000),weighting=function(x) weightTf(x)));
        }
        if(do.trace) print("From sparse to dense matrix");
        m = dtm;
        
        
        #Cut-off threshold
        
        cs = col_sums(m>0);
        n_doc = dim(m)[1];
        if(do.trace) print("Removing sparse terms");
        
        # drop the terms with less than min_df freq
        
        if (min_df %% 1 == 0){#Is it an integer?
                m = m[,cs>=min_df];
                cs = col_sums(m>0);
                
                if(do.trace) print(paste("Dimensions:",dim(m)[1],":",dim(m)[2]));
        } else {
                #Is it a percentage?
                if (min_df > 0 && min_df <= 1){
                        thr = n_doc * (1-min_df);
                        if (thr < 1) thr = 1;
                        m = m[,cs>=thr];
                        cs = col_sums(m>0);
                } else {
                        stop("Error : threshold is neither an integer nor a percentage");
                }
        }
        
        # Sublinear term-frequency scaling, i.e. replace tf with 1 + log(tf).
        if(sublinear_tf==TRUE) {
                if(do.trace) print("Applying sublinear tf scaling!");
                m$v = 1 + log(m$v);
        }
        
        # Smoothing weights by adding one to document frequencies, as if an
        # extra document was seen containing every term in the collection
        # exactly once. Prevents zero divisions.
        if(smooth_idf==TRUE) {
                if(do.trace) print("Smoothing!");
                n_doc = n_doc + 1;
                cs = cs + 1;
        }
        
        # Cast to sparse matrix
        # so that Diagonal * m is fast and eficient
        m = sparseMatrix(m$i,m$j,x=m$v,dims=dim(m),dimnames=dimnames(m));
        if (use_idf){
                idf = 1+log(n_doc/cs);
                d = Diagonal(length(idf),idf);
                m = m%*%d;
                d = NULL;
        }
        
        if (is.null(normf)) normf="";
        
        # Normalize L1 or L2 
        if (!is.na(match(normf,c("l1","l2")))) {
                if(do.trace) print(paste("Applying",normf,"normalization!"));
                l_m = m;
                if (normf=="l2"){
                        l_m@x = l_m@x^2;
                        rs = sqrt(rowSums(l_m));
                } else {
                        l_m@x = abs(l_m@x);
                        rs = rowSums(l_m);
                }
                #avoid division by zero
                rs[rs==0] = 1;
                m = m / rs;
        }
        
        # return sparse matrix
        if(do.trace) print("Done!");
        return(m);
}




#Matching variable
similarity<-function(x){
        
        terms1<-x[1]
        terms2<-x[2]
        
        corpus<-Corpus(VectorSource(list(terms1,terms2)))
        corpus<-tm_map(corpus,content_transformer(function(x)iconv(enc2utf8(x),sub="byte")))
        corpus<-tm_map(corpus,content_transformer(tolower))
        corpus<-tm_map(corpus,removePunctuation)
        corpus<-tm_map(corpus,removeWords,dict)
        corpus<-tm_map(corpus,stemDocument)
        corpus<-tm_map(corpus,stripWhitespace)
        freq<-DocumentTermMatrix(corpus)
        freq<-as.matrix(freq)
        similarity<-sum(freq[1,]>0&freq[2,]>0)
        
        return(similarity)               
}

#Distance function : cosine between query and results

distance<-function(x){
        
        terms1<-x[1]
        terms2<-x[2]
        
        corpus<-Corpus(VectorSource(list(terms1,terms2)))
        corpus<-tm_map(corpus,content_transformer(function(x)iconv(enc2utf8(x),sub="byte")))
        corpus<-tm_map(corpus,content_transformer(tolower))
        corpus<-tm_map(corpus,removePunctuation)
        corpus<-tm_map(corpus,removeWords,dict)
        corpus<-tm_map(corpus,stemDocument)
        corpus<-tm_map(corpus,stripWhitespace)
        dist<-DocumentTermMatrix(corpus)
        dist<-as.matrix(dist)
        distance<-cosine(dist[1,],dist[2,])
        
        return(distance)       
}


train = read_csv("train.csv");
test = read_csv("test.csv");

y = train$median_relevance;
y_var = train$relevance_variance;
queries = as.factor(train$query);

# combine query title and description into single character array
txt = paste(train$query,train$product_title);
txt = c(txt,paste(test$query,test$product_title));

#Similarity and distance
data1<-select(train,query,product_title)
data2<-select(test,query,product_title)
data1<-rbind(data1,data2)

similarity<-apply(data1[,c(1,2)],1,similarity)
distance<-apply(data1[,c(1,2)],1,distance)
distance[is.na(distance)]<-0
metr<-cbind(similarity,distance,manhattan,dist_euclid)

rm(data2,data1)



# get document term matrix
m = tfidf(txt,sublinear=T,smooth_idf=T,normf="l2",min_df=3);

# split to train and test
train.data = m[1:nrow(train),];
test.data = m[(nrow(train)+1):nrow(m),];
train.metr<-metr[1:nrow(train),]
test.metr<-metr[-c(1:nrow(train)),]

# cv_fold_count tells the script how much 
# folds to use to estimate kappa metric
cv_fold_count = 3;

# parallel infrastructure
n.cores = detectCores();
cl <- makeCluster(n.cores); 
registerDoParallel(cl);

# tuning grid, change ncomp and cost
tune.grid = expand.grid(ncomp=c(400,450),cost=c(12,15));

# run grid search in parallel
results = 
        foreach(gridId = c(1:nrow(tune.grid)), .packages=c('kernlab','rARPACK','caret','Metrics','Matrix'),
                .combine=rbind, .multicombine=T) %dopar% {
                        set.seed(2603); #-> so it can be compared when using similar comps, e.g. 10,11,12
                        
                        # stratified folds by queries so each fold has approx same query dist.
                        folds = createFolds(as.factor(y),cv_fold_count);
                        svm_cost = tune.grid[gridId,"cost"];
                        svd_ncomp = tune.grid[gridId,"ncomp"];
                        q.kappa = 0;
                        
                        # do the folds
                        for(i in 1: length(folds)){
                                # get the sampling
                                # and construct train and test matrix out of train data
                                smpl = folds[[i]];
                                g_train = train.data[-smpl,];
                                g_test = train.data[smpl,];
                                y_train = y[-smpl];
                                y_test = y[smpl];
                                train_metr_cv<-train.metr[-smpl,]
                                test_metr_cv<-train.metr[smpl,]
                                
                                #svd here
                                g_train_svd<-svds(g_train,k=svd_ncomp,nv=svd_ncomp,nu=0)
                                
                                # note that u must multiply svd$v matrix with train and 
                                # test matrix.
                                g_train = g_train%*%g_train_svd$v;
                                g_test = g_test%*%g_train_svd$v;
                                
                                
                                #Include distance merics
                                g_train<-cbind(g_train,train_metr_cv)
                                g_test<-cbind(g_test,test_metr_cv)
                                
                                # train the svm, I'm using kernlab but e1071 svm is also good
                                # first one seems to give better results for same hyper params
                                sv = ksvm(as.matrix(g_train),as.factor(y_train),kernel="rbfdot",
                                          C=svm_cost,scaled=T,kpar=list("sigma"=1/dim(g_train)[2]));
                                p = predict(sv,newdata=g_test,type="response");
                                
                                # calc the quadratic kappa
                                q.kappa = q.kappa + ScoreQuadraticWeightedKappa(as.numeric(as.character(y_test)),
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

best_result = (results[order(results$qkapp,decreasing=T),])[1,];
best_result;

# svd 
g_train_svd = svds(train.data,k = best_result$svd_ncomp,nv =best_result$svd_ncomp, nu=0);

train.data.fin = train.data %*% g_train_svd$v;
train.data.fin<-cbind(train.data.fin,train.metr)
test.data.fin = test.data %*% g_train_svd$v;
test.data.fin<-cbind(test.data.fin,test.metr)

# train best model
sv_model = ksvm(as.matrix(train.data.fin),as.factor(y),
                kernel="rbfdot",
                C=best_result$svm_cost,
                scaled=T,
                kpar=list("sigma"=1/dim(train.data.fin)[2]));

#submission
sub.p = predict(sv_model,newdata=test.data.fin,type="response");

table(sub.p);

write.csv(data.frame("id"=test$id, "prediction"=sub.p),"submission2.csv",quote=F,row.names=F);


