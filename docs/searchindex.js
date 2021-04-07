Search.setIndex({docnames:["Distillation/callbacks","Distillation/data","Distillation/distillation","Distillation/loss_functions","Distillation/runners","Examples/classification_huggingface_transformers","Models/cv","Pruning/pruning","Quantization/quantization","examples","index","models","runners"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["Distillation/callbacks.rst","Distillation/data.rst","Distillation/distillation.rst","Distillation/loss_functions.rst","Distillation/runners.rst","Examples/classification_huggingface_transformers.rst","Models/cv.rst","Pruning/pruning.rst","Quantization/quantization.rst","examples.rst","index.rst","models.rst","runners.rst"],objects:{"compressors.distillation":{losses:[3,1,0,"-"],runners:[4,1,0,"-"]},"compressors.distillation.callbacks":{AttentionHiddenStatesCallback:[2,0,1,""],HiddenStatesSelectCallback:[0,0,1,""],KLDivCallback:[2,0,1,""],LambdaPreprocessCallback:[0,0,1,""],MSEHiddenStatesCallback:[2,0,1,""],hidden_states:[0,1,0,"-"],logits_diff:[0,1,0,"-"],wrappers:[0,1,0,"-"]},"compressors.distillation.callbacks.hidden_states":{AttentionHiddenStatesCallback:[0,0,1,""],CosineHiddenStatesCallback:[0,0,1,""],HiddenStatesSelectCallback:[0,0,1,""],LambdaPreprocessCallback:[0,0,1,""],MSEHiddenStatesCallback:[0,0,1,""],PKTHiddenStatesCallback:[0,0,1,""]},"compressors.distillation.callbacks.logits_diff":{KLDivCallback:[0,0,1,""]},"compressors.distillation.callbacks.wrappers":{LambdaWrapperCallback:[0,0,1,""]},"compressors.distillation.data":{LogitsDataset:[1,0,1,""],label_smoothing:[1,1,0,"-"]},"compressors.distillation.data.label_smoothing":{probability_shift:[1,2,1,""]},"compressors.distillation.losses":{AttentionLoss:[3,0,1,""],CRDLoss:[3,0,1,""],KLDivLoss:[3,0,1,""],MSEHiddenStatesLoss:[3,0,1,""],kl_div_loss:[3,2,1,""],mse_loss:[3,2,1,""],pkt_loss:[3,2,1,""]},"compressors.distillation.losses.AttentionLoss":{forward:[3,3,1,""]},"compressors.distillation.losses.CRDLoss":{forward:[3,3,1,""]},"compressors.distillation.losses.KLDivLoss":{forward:[3,3,1,""]},"compressors.distillation.losses.MSEHiddenStatesLoss":{forward:[3,3,1,""]},"compressors.distillation.losses._kl_loss":{KLDivLoss:[2,0,1,""]},"compressors.distillation.runners":{DistilRunner:[4,0,1,""],EndToEndDistilRunner:[4,0,1,""],HFDistilRunner:[4,0,1,""]},"compressors.distillation.runners.EndToEndDistilRunner":{get_callbacks:[4,3,1,""],get_stage_len:[4,3,1,""],predict_batch:[4,3,1,""],stages:[4,3,1,""]},"compressors.models":{base_distil_model:[11,1,0,"-"],cv:[6,1,0,"-"]},"compressors.models.base_distil_model":{BaseDistilModel:[11,0,1,""]},"compressors.models.base_distil_model.BaseDistilModel":{forward:[11,3,1,""]},"compressors.models.cv":{resnet101:[6,2,1,""],resnet152:[6,2,1,""],resnet18:[6,2,1,""],resnet34:[6,2,1,""],resnet50:[6,2,1,""],resnet_cifar_110:[6,2,1,""],resnet_cifar_14:[6,2,1,""],resnet_cifar_20:[6,2,1,""],resnet_cifar_32:[6,2,1,""],resnet_cifar_32x4:[6,2,1,""],resnet_cifar_44:[6,2,1,""],resnet_cifar_56:[6,2,1,""],resnet_cifar_8:[6,2,1,""],resnet_cifar_8x4:[6,2,1,""],resnext101_32x8d:[6,2,1,""],resnext50_32x4d:[6,2,1,""],wide_resnet101_2:[6,2,1,""],wide_resnet50_2:[6,2,1,""]},"compressors.pruning":{callbacks:[7,1,0,"-"],runners:[7,1,0,"-"],utils:[7,1,0,"-"]},"compressors.pruning.callbacks":{LotteryTicketCallback:[7,0,1,""],PrepareForFinePruningCallback:[7,0,1,""]},"compressors.pruning.callbacks.LotteryTicketCallback":{on_stage_start:[7,3,1,""]},"compressors.pruning.callbacks.PrepareForFinePruningCallback":{on_experiment_start:[7,3,1,""]},"compressors.pruning.runners":{FinePruneRunner:[7,0,1,""],PruneRunner:[7,0,1,""]},"compressors.pruning.runners.FinePruneRunner":{get_callbacks:[7,3,1,""],get_stage_len:[7,3,1,""],stages:[7,3,1,""]},"compressors.pruning.runners.PruneRunner":{handle_batch:[7,3,1,""],predict_batch:[7,3,1,""],stages:[7,3,1,""]},"compressors.runners":{HFRunner:[12,0,1,""]},"compressors.runners.HFRunner":{handle_batch:[12,3,1,""]},compressors:{runners:[12,1,0,"-"]}},objnames:{"0":["py","class","Python class"],"1":["py","module","Python module"],"2":["py","function","Python function"],"3":["py","method","Python method"]},objtypes:{"0":"py:class","1":"py:module","2":"py:function","3":"py:method"},terms:{"07471":1,"101":6,"1024":6,"10837":0,"110":6,"128":[3,5,9],"128_a":5,"152":6,"16384":3,"1803":0,"1911":1,"1x1":6,"2048":6,"2_h":5,"32x4":6,"32x4d":6,"32x8d":6,"4_h":5,"512":[5,6],"512_a":5,"8x4":6,"case":9,"class":[0,1,2,3,4,7,9,11,12],"default":[0,2,3,4,11],"final":[4,9],"float":[0,2,3],"function":[0,1,2,10],"import":[5,9],"int":[0,2,3,4,7,9],"return":[1,3,4,5,6,7,9,11],"true":[0,1,2,4,5,6,9,11],"while":3,But:[2,9],For:[1,2],The:[2,3,6,9],There:10,Used:[7,12],Useful:0,__init__:9,_kl_loss:2,abs:[0,1],accumul:9,accuraci:5,accuracy01:9,accuracycallback:9,adam:[5,9],add:[2,9,11],add_batch:5,addit:[4,7,9],afterward:3,ag_new:5,aggreg:6,aka:[0,2],all:[3,9],also:[2,9],although:3,analog:0,anchor:3,ani:[4,6,7,12],api:10,append:9,appli:[0,2],apply_probability_shift:4,approach:2,arg:[4,9,11,12],argmax:[1,5],arrai:[4,7],arxiv:[0,1],attent:[0,2],attention_loss:[0,2],attention_mask:5,attentionhiddenstatescallback:[0,2],attentionloss:[2,3],automodelforsequenceclassif:5,autonotebook:5,autotoken:5,backward:5,bar:6,base:[0,1,11],base_callback:0,base_distil_model:11,basedistilmodel:[9,11],batch:[0,1,4,5,7,12],batch_siz:[1,3,5,9],befor:0,begin:9,bert_uncased_l:5,best_accuraci:5,best_stud:5,better:[1,2],between:[0,2,3,4,9],big:2,block:6,bool:[0,1,2,3,4,6,9,11],bottleneck:6,buffer:3,call:[2,3],callabl:[0,1,4],callback:[2,4,9,10],can:[2,4,9,10],care:3,catalyst:[0,2,4,7,9,10],cdot:9,chain:9,channel:6,choos:3,cifar:6,classic:2,classif:[0,2],code:9,collect:7,column:5,common:2,complex:[2,10],compress:10,compressor:[0,1,2,3,4,5,6,7,9,11,12],comput:[3,5,10,11],content:10,contrast:3,contrast_idx:3,contrib:9,convolut:6,core:[0,4,7],correct:1,cosin:[0,3],cosine_loss:0,cosinehiddenstatescallback:0,count:9,cpu:5,crd:3,crdloss:3,creat:9,criterion:9,criterioncallback:9,crossentropyloss:9,cuda:5,cur_hidden:9,data:[2,4,5,7,9,10,12],data_loader_kwarg:1,dataload:[1,4,5,7,9,12],dataset:[1,3,5,9],deep:6,def:[5,9],defin:[3,9],detach:5,deverg:3,devic:5,dict:[0,4,7,9,11],dict_to_devic:5,dictionari:[4,7,12],differ:[2,9],dimens:3,displai:6,distil:[0,1,3,4,5,9,10,11],distilrunn:[2,4,9],distribut:[0,2,3,9],diverg:[2,3,9],dl_kei:9,doesn:[0,2],don:0,download:[6,9],dure:[7,12],each:3,either:10,elif:9,els:[5,9,11],embed:3,end:4,endtoenddistilrunn:[4,9],enumer:9,epoch:[4,5,7],eps:3,eval:5,event:7,everi:[3,6],exampl:[2,10],examplemodel:9,exapml:1,except:6,exclude_first_and_last:[0,2],experi:[2,4,7,12],f_s:3,f_t:3,fals:[0,2,3,4,5,6,9,11],feat_dim:3,featur:[3,7],feature_dim:3,filter:0,final_loss:5,fineprunerunn:7,first:[0,2,9],flag:1,floattensor:3,forc:2,former:3,forward:[3,9,11],frac:2,from:[1,4,5,6,7,9,12],from_pretrain:5,get:1,get_callback:[4,7],get_logits_fn:1,get_stage_len:[4,7],given:[4,7],googl:5,handl:[7,12],handle_batch:[7,12],handler:7,has:[6,9],have:2,here:[2,9],hfdistilrunn:4,hfrunner:12,hidden:[2,3,4,9,11],hidden_dim:9,hidden_st:[0,5,9],hidden_state_loss:[4,5,9],hiddens_criterion:5,hiddens_kei:0,hiddens_loss:5,hiddenstatesselectcallback:[0,9],hint:[0,2],hook:3,http:[0,1],huggingfac:[9,10,12],idea:10,idx:3,ignor:3,imag:[2,6],imagenet:6,implement:[2,4,7],improv:1,includ:3,indic:3,infer:[4,7,12],initi:9,initial_state_dict:7,inner:[7,12],inp:9,inplac:0,input:0,input_id:5,input_kei:[7,9],instanc:[3,7],instead:3,irunn:7,is_avail:5,item:[5,9],iten:5,itertool:9,join:5,just:[2,4,9,10],kei:[0,5],kernel:3,keys_to_appli:0,kl_callback:9,kl_div_loss:[0,2,3,5],kl_loss:[4,9],kldivcallback:[0,2,9],kldivloss:[2,3,5,9],knoweledg:[2,11],knowledg:[1,2,4],knowlewdg:0,kwarg:[1,4,6,7,11,12],l_2:2,label:[1,5],label_smooth:1,lambda:[0,5],lambda_fn:0,lambdapreprocesscallback:0,lambdawrapp:0,lambdawrappercallback:0,larger:6,last:[0,2,4,6,9],last_onli:0,latter:3,layer:[0,9],layer_idx:9,learn:[6,9],leav:5,left:2,length:[4,7],lesson:1,let:[2,9],librari:10,like:2,linear:9,list:[0,7],load:5,load_dataset:5,load_metr:5,load_state_dict:5,loader:[5,9],loader_kei:5,log:[2,9],log_str:5,logdir:9,logit:[1,2,4,5,7,9],logits_criterion:5,logits_dataset_arg:7,logits_dataset_kwarg:7,logits_diff:0,logits_diff_loss:4,logits_loss:5,logitsdataset:1,look:2,loss:[0,2,4,5,9,10],loss_weight:4,lotteryticketcallback:7,main:[9,10],make:[7,12],mani:10,map:[4,5,7,11,12],mapping_optim:5,max_length:5,measur:2,memori:3,merg:1,merge_logits_with_batch_fn:1,met:5,method:[2,7,11,12],metric:5,metric_fn:5,metricaggregationcallback:9,minim:10,minimize_valid_metr:9,mlp:9,mnist:9,model:[0,1,2,3,4,6,7,9,10],modifi:0,modul:[1,3],modulelist:9,momentum:3,more:[2,9],most:2,mse:[0,2,3,9],mse_callback:9,mse_hiddens_loss:5,mse_loss:[0,2,3,9],msehiddenstatescallback:[0,2,9],msehiddenstatesloss:[2,3,5],mseloss:9,n_data:3,name:[0,2,4,7],nce_k:3,nce_m:3,nce_t:3,need:[2,3,9],need_map:[0,2,3,5],neg:3,network:[2,3,6],neural:[2,6],nlp:10,no_grad:5,none:[0,1,2,3,4,7],norm:2,normal:[0,2,3,5],notimplementederror:[4,7],now:9,num_epoch:[5,9],num_label:5,num_lay:[0,2,3,5,9],num_sess:7,num_train_teacher_epoch:[4,9],number:[3,4,6],numpi:5,on_experiment_start:7,on_stage_start:7,one:[2,3,4,7],onli:[0,2,7,9],opt:3,optim:[5,9],optimi:9,option:[0,1,2,3,4,11],ordereddict:[4,7],ordinari:9,org:[0,1],other:2,our:9,outer:6,output:[0,3,4,7,9,11],output_hidden_st:[4,5,9,11],output_kei:[0,2,7],over:[0,3],overridden:3,p_1:2,p_2:2,pad:5,pair:3,paper:2,param:6,paramet:[0,1,2,3,4,5,7,9,11,12],part:[3,10],pass:[3,4,7],pbar_epoch:5,pbar_load:5,perform:[2,3],philosophi:10,pipelin:9,pkt:0,pkt_loss:[0,3],pkthiddenstatescallback:0,posit:3,pre:6,predict:5,predict_batch:[4,7],prepar:1,prepareforfinepruningcallback:7,preprocess:[0,9],preprocessor:2,pretrain:6,print:5,probabilist:0,probability_shift:1,probabl:[0,2,9],produc:2,progress:6,project:3,properti:[4,7],propos:[0,2],provid:[4,9,10],prune:10,prunerunn:7,pth:5,pytorch:9,quantiz:10,rais:[0,4,7],rang:9,readi:9,recip:3,recognit:6,refer:5,regist:3,relu:9,residu:6,resnet101:6,resnet152:6,resnet18:6,resnet34:6,resnet50:6,resnet:6,resnet_cifar:6,resnet_cifar_110:6,resnet_cifar_14:6,resnet_cifar_20:6,resnet_cifar_32:6,resnet_cifar_32x4:6,resnet_cifar_44:6,resnet_cifar_56:6,resnet_cifar_8:6,resnet_cifar_8x4:6,resnetcifar:6,resnext101_32x8d:6,resnext50_32x4d:6,resnext:6,retrun_dict:5,return_dict:[9,11],right:2,run:[3,4,7,9,11,12],runner:[0,2,9,10],runner_arg:[4,7],runner_kwarg:[4,7],s_dim:3,s_h:5,s_hidden:5,s_hidden_st:[0,3],s_logit:[0,2,3,9],same:[2,6],sampl:3,save:5,select_last_hidden_st:9,self:9,separ:10,set:[0,2,3,4,9],set_descript:5,set_format:5,sever:9,shape:2,should:[3,9],show:2,shuffl:[5,9],side:3,silent:3,similar:3,simpl:[2,4,9],simpli:9,sinc:3,size:3,slow:2,small:3,smaller:2,smooth:1,someth:[0,2],space:3,specifi:[0,4,7,12],stage:[4,7,12],start:7,startswith:5,state:[2,3,4,9,11],state_dict:5,stderr:6,step:5,str:[0,2,4,7,12],student:[0,2,3,5,9],student_dim:3,student_hidden_state_dim:[0,2,3,5],student_logits_kei:[0,2],student_model:5,student_output:5,subclass:3,submodul:10,sum:[4,9],sum_x:2,supervis:[1,9,12],supervisedrunn:9,support:9,swap:1,symmetr:3,t_dim:3,t_h:5,t_hidden:5,t_hidden_st:[0,3],t_logit:[0,2,3],take:[0,1,2,3],target:9,target_kei:9,task:[0,2,9,10,12],task_loss:5,teacher:[0,1,2,3,4,5,9],teacher_dim:3,teacher_hidden_state_dim:[0,2,3,5],teacher_logits_kei:[0,2],teacher_model:5,teacher_output:5,temperatur:[0,2,3],tensor:[1,2],test:5,text:[5,9],them:3,therefor:[3,9],thi:[0,2,3,7,9],three:10,time:2,tini:9,token:5,token_type_id:5,took:[2,9],torch:[1,3,5,9],torchvisiondatasetwrapp:9,tqdm:5,train:[1,2,3,4,5,6,7,9,12],train_it:5,train_loader_kei:7,trang:5,transfer:[0,2],transform:[4,6,9,10,12],truncat:5,tupl:[3,5,9,11],twice:6,two:[3,9,10],type:[3,4,5,6,7,9,11],typeerror:0,union:[0,3,4,7,11],updat:3,use:[0,2,10],used:4,useful:0,using:[2,3,9],usual:[0,2],util:[1,5,9,10],val_it:5,valid:[5,7,9,12],valid_metr:9,valu:[3,4,7],vector:2,version:6,vision:[10,11],w_1:9,w_2:9,w_3:9,wai:[0,2,10],want:0,weight:[4,5,9],when:0,which:[0,6],wide:6,wide_resnet101_2:6,wide_resnet50_2:6,within:3,without:4,work:7,wrap:0,wrapper:[1,2],wrp:9,yet:[4,7],you:[0,2,10],your:[0,10],zero_grad:5},titles:["Callbacks","Data","Distillation","Loss functions","Runners","Huggingface transformers","Computer Vision","Pruning","Quantization","Examples","Welcome to Compressors\u2019s documentation!","Models","Runners"],titleterms:{"function":3,callback:[0,7],complex:9,compressor:10,comput:6,data:1,differ:0,distil:2,document:10,exampl:9,hidden:0,huggingfac:5,idea:2,logit:0,loss:3,main:2,minim:9,model:11,nlp:9,philosophi:2,preprocessor:0,prune:7,quantiz:8,runner:[4,7,12],state:0,submodul:2,transform:5,util:7,vision:6,welcom:10,why:10,wrapper:0}})