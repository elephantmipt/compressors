Search.setIndex({docnames:["Distillation/callbacks","Distillation/data","Distillation/distillation","Distillation/loss_functions","Distillation/runners","Examples/classification_huggingface_transformers","Models/cv","Pruning/pruning","Quantization/quantization","examples","index","models","runners"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["Distillation/callbacks.rst","Distillation/data.rst","Distillation/distillation.rst","Distillation/loss_functions.rst","Distillation/runners.rst","Examples/classification_huggingface_transformers.rst","Models/cv.rst","Pruning/pruning.rst","Quantization/quantization.rst","examples.rst","index.rst","models.rst","runners.rst"],objects:{"compressors.distillation":{losses:[3,0,0,"-"],runners:[4,0,0,"-"]},"compressors.distillation.callbacks":{hidden_states:[0,0,0,"-"],logits_diff:[0,0,0,"-"],wrappers:[0,0,0,"-"]},"compressors.distillation.callbacks.hidden_states":{AttentionHiddenStatesCallback:[0,1,1,""],CosineHiddenStatesCallback:[0,1,1,""],HiddenStatesSelectCallback:[0,1,1,""],LambdaSelectCallback:[0,1,1,""],MSEHiddenStatesCallback:[0,1,1,""],PKTHiddenStatesCallback:[0,1,1,""]},"compressors.distillation.callbacks.logits_diff":{KLDivCallback:[0,1,1,""]},"compressors.distillation.callbacks.wrappers":{LambdaWrp:[0,1,1,""]},"compressors.distillation.data":{LogitsDataset:[1,1,1,""],swap_smoothing:[1,1,1,""]},"compressors.distillation.losses":{AttentionLoss:[3,1,1,""],CRDLoss:[3,1,1,""],KLDivLoss:[3,1,1,""],MSEHiddenStatesLoss:[3,1,1,""],kl_div_loss:[3,3,1,""],mse_loss:[3,3,1,""],pkt_loss:[3,3,1,""]},"compressors.distillation.losses.AttentionLoss":{forward:[3,2,1,""]},"compressors.distillation.losses.CRDLoss":{forward:[3,2,1,""]},"compressors.distillation.losses.KLDivLoss":{forward:[3,2,1,""]},"compressors.distillation.losses.MSEHiddenStatesLoss":{forward:[3,2,1,""]},"compressors.distillation.runners":{DistilRunner:[4,1,1,""],EndToEndDistilRunner:[4,1,1,""],HFDistilRunner:[4,1,1,""]},"compressors.distillation.runners.EndToEndDistilRunner":{get_callbacks:[4,2,1,""],get_stage_len:[4,2,1,""],predict_batch:[4,2,1,""],stages:[4,2,1,""]},"compressors.models":{base_distil_model:[11,0,0,"-"]},"compressors.models.base_distil_model":{BaseDistilModel:[11,1,1,""]},"compressors.models.base_distil_model.BaseDistilModel":{forward:[11,2,1,""]},"compressors.models.cv":{preact_resnet:[6,0,0,"-"],resnet:[6,0,0,"-"],resnet_cifar:[6,0,0,"-"]},"compressors.models.cv.resnet":{ResNet:[6,1,1,""],conv1x1:[6,3,1,""],resnet101:[6,3,1,""],resnet152:[6,3,1,""],resnet18:[6,3,1,""],resnet34:[6,3,1,""],resnet50:[6,3,1,""],resnext101_32x8d:[6,3,1,""],resnext50_32x4d:[6,3,1,""],wide_resnet101_2:[6,3,1,""],wide_resnet50_2:[6,3,1,""]},"compressors.models.cv.resnet.ResNet":{forward:[6,2,1,""]},"compressors.models.cv.resnet_cifar":{ResNetCifar:[6,1,1,""]},"compressors.models.cv.resnet_cifar.ResNetCifar":{forward:[6,2,1,""]},"compressors.runners":{HFRunner:[12,1,1,""]},"compressors.runners.HFRunner":{handle_batch:[12,2,1,""]},compressors:{runners:[12,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"07471":1,"1000":6,"101":6,"1024":6,"10837":0,"128":[3,5,9],"128_a":5,"152":6,"16384":3,"1803":0,"1911":1,"1x1":6,"2048":6,"2_h":5,"32x4d":6,"32x8d":6,"4_h":5,"512":[5,6],"512_a":5,"case":9,"class":[0,1,3,4,6,9,11,12],"default":[0,3,4,6,11],"final":[4,9],"float":3,"function":[0,1,2,10],"import":[5,9],"int":[0,3,4,6,9],"return":[1,3,4,5,6,9,11],"true":[0,1,4,5,6,9,11],"while":3,But:9,For:1,The:[3,6,9],There:10,Used:12,Useful:0,__init__:9,abs:[0,1],accumul:9,accuraci:5,accuracy01:9,accuracycallback:9,adam:[5,9],adapt:6,add:[6,9,11],add_batch:5,addit:[4,9],afterward:3,ag_new:5,aggreg:6,aka:0,all:[3,9],also:9,although:3,analog:0,anchor:3,ani:[4,6,12],api:10,append:9,appli:0,arg:[4,9,11,12],argmax:[1,5],arrai:4,arxiv:[0,1],attention_loss:0,attention_mask:5,attentionhiddenstatescallback:0,attentionloss:3,automodelforsequenceclassif:5,autonotebook:5,autotoken:5,backward:5,bar:6,base:[0,1,11],base_callback:0,base_distil_model:11,basedistilmodel:[9,11],basicblock:6,batch:[0,1,4,5,12],batch_siz:[1,3,5,9],begin:9,bert_uncased_l:5,best_accuraci:5,best_stud:5,better:1,between:[0,3,4,9],blob:6,block:6,block_nam:6,bool:[0,1,3,4,6,9,11],bottleneck:6,buffer:3,call:3,callabl:[0,1,4,6],callback:[2,4,9,10],can:[4,9,10],care:3,catalyst:[0,4,9,10],cdot:9,chain:9,channel:6,choos:3,classif:0,code:9,column:5,com:6,complex:10,compress:10,compressor:[0,1,3,4,5,6,9,11,12],comput:[3,5,10,11],content:10,contrast:3,contrast_idx:3,contrib:9,conv1x1:6,conv2d:6,conv:6,convolut:6,core:[0,4],correct:1,cosin:[0,3],cosine_loss:0,cosinehiddenstatescallback:0,count:9,cpu:5,crd:3,crdloss:3,creat:9,criterion:9,criterioncallback:9,crossentropyloss:9,cuda:5,cur_hidden:9,data:[2,4,5,9,10,12],data_loader_kwarg:1,dataload:[1,4,5,9,12],dataset:[1,3,5,9],deep:6,def:[5,9],defin:[3,9],depth:6,detach:5,deverg:3,devic:5,dict:[0,4,6,9,11],dict_to_devic:5,dictionari:[4,12],differ:[2,9],dimens:3,displai:6,distil:[0,1,3,4,5,9,10,11],distilrunn:[4,9],distribut:[0,3,9],diverg:[3,9],dl_kei:9,don:0,download:[6,9],dure:12,each:3,either:10,elif:9,els:[5,6,9,11],embed:3,end:4,endtoenddistilrunn:[4,9],enumer:9,epoch:[4,5],eps:3,eval:5,everi:[3,6],exampl:10,examplemodel:9,exapml:1,except:6,exclude_first_and_last:0,experi:[4,12],f_s:3,f_t:3,fals:[0,3,5,6,9,11],feat_dim:3,featur:3,feature_dim:3,filter:0,final_loss:5,first:9,flag:1,floattensor:[3,6],former:3,forward:[3,6,9,11],from:[1,4,5,6,9,12],from_pretrain:5,get:1,get_callback:4,get_logits_fn:1,get_stage_len:4,github:6,given:4,googl:5,group:6,handl:12,handle_batch:12,has:[6,9],here:9,hfdistilrunn:4,hfrunner:12,hidden:[2,3,4,6,9,11],hidden_dim:9,hidden_st:[0,5,9],hidden_state_loss:[4,5,9],hiddens_criterion:5,hiddens_kei:0,hiddens_loss:5,hiddenstatesselectcallback:[0,9],hint:0,hook:3,http:[0,1,6],huggingfac:[9,12],idx:3,ignor:3,imag:6,imagenet:6,implement:4,improv:1,in_plan:6,includ:3,indic:3,infer:[4,12],initi:9,inner:12,inp:9,inplac:0,input:0,input_id:5,input_kei:9,instanc:3,instead:3,is_avail:5,item:[5,9],iten:5,itertool:9,join:5,just:[4,9,10],kei:[0,5],kernel:3,keys_to_appli:0,kl_callback:9,kl_div_loss:[0,3,5],kl_loss:[4,9],kldivcallback:[0,9],kldivloss:[3,5,9],knoweledg:11,knowledg:[1,4],knowlewdg:0,kwarg:[1,4,6,11,12],label:[1,5],lambda:[0,5],lambda_fn:0,lambdaselectcallback:0,lambdawrp:0,larger:6,last:[0,4,6,9],last_onli:0,latter:3,layer:[0,6,9],layer_idx:9,learn:[6,9],leav:5,length:4,lesson:1,let:9,librari:10,linear:9,list:[0,6],load:5,load_dataset:5,load_metr:5,load_state_dict:5,loader:[5,9],loader_kei:5,log:9,log_str:5,logdir:9,logit:[1,2,4,5,9],logits_criterion:5,logits_diff:0,logits_diff_loss:4,logits_loss:5,logitsdataset:1,loss:[0,2,4,5,9,10],loss_weight:4,main:9,make:12,mani:10,map:[4,5,6,11,12],mapping_optim:5,master:6,max_length:5,memori:3,merg:1,merge_logits_with_batch_fn:1,met:5,method:[6,11,12],metric:5,metric_fn:5,metricaggregationcallback:9,minim:10,minimize_valid_metr:9,mlp:9,mnist:9,model:[0,1,3,4,6,9,10],modifi:0,modul:[1,3,6],modulelist:9,momentum:3,more:9,mse:[0,3,9],mse_callback:9,mse_hiddens_loss:5,mse_loss:[0,3,9],msehiddenstatescallback:[0,9],msehiddenstatesloss:[3,5],mseloss:9,n_data:3,name:[0,4],nce_k:3,nce_m:3,nce_t:3,need:[3,9],need_map:[0,3,5],neg:3,network:[3,6],neural:6,nlp:10,no_grad:5,none:[0,1,3,4,6],norm_lay:6,normal:[0,3,5],notimplementederror:4,now:9,num_class:6,num_epoch:[5,9],num_filt:6,num_label:5,num_lay:[0,3,5,9],num_train_teacher_epoch:[4,9],number:[3,4,6],numpi:5,one:[3,4],onli:[0,9],opt:3,optim:[5,9],optimi:9,option:[0,1,3,4,6,11],ordereddict:4,ordinari:9,org:[0,1],our:9,out_plan:6,outer:6,output:[0,3,4,6,9,11],output_hidden_st:[4,5,6,9,11],output_kei:0,over:[0,3],overridden:3,pad:5,pair:3,param:[1,3,6],paramet:[0,1,3,4,5,6,9,11,12],part:[3,10],pass:[3,4],pbar_epoch:5,pbar_load:5,perform:3,pipelin:9,pkt:0,pkt_loss:[0,3],pkthiddenstatescallback:0,posit:3,pre:6,preact:6,predict:5,predict_batch:4,prepar:1,preprocess:9,pretrain:6,print:5,probabilist:0,probabl:[0,9],progress:6,project:3,properti:4,propos:0,provid:[4,9,10],prune:10,pth:5,pytorch:[6,9],quantiz:10,rais:[0,4],rang:9,readi:9,recip:3,recognit:6,refer:5,regist:3,relu:9,replace_stride_with_dil:6,residu:6,resnet101:6,resnet152:6,resnet18:6,resnet34:6,resnet50:6,resnet:6,resnet_cifar:6,resnet_modul:6,resnetcifar:6,resnext101_32x8d:6,resnext50_32x4d:6,resnext:6,retrun_dict:5,return_dict:[6,9,11],run:[3,4,9,11,12],runner:[0,2,9,10,12],runner_arg:4,runner_kwarg:4,s_dim:3,s_h:5,s_hidden:5,s_hidden_st:[0,3],s_logit:[3,9],same:6,sampl:3,save:5,select_last_hidden_st:9,self:9,separ:10,set:[0,3,4,9],set_descript:5,set_format:5,sever:9,should:[3,9],shuffl:[5,9],side:3,silent:3,similar:3,simpl:[4,9],simpli:9,sinc:3,size:3,small:3,smooth:1,someth:0,space:3,specifi:[0,4,12],stage:[4,12],startswith:5,state:[2,3,4,6,9,11],state_dict:5,stderr:6,step:5,str:[0,4,12],stride:6,student:[0,3,5,9],student_dim:3,student_hidden_state_dim:[0,3,5],student_model:5,student_output:5,subclass:3,sum:[4,9],supervis:[1,9,12],supervisedrunn:9,support:9,swap:1,swap_smooth:1,symmetr:3,t_dim:3,t_h:5,t_hidden:5,t_hidden_st:[0,3],t_logit:3,take:[0,1,3],target:9,target_kei:9,task:[0,9,10,12],task_loss:5,teacher:[0,1,3,4,5,9],teacher_dim:3,teacher_hidden_state_dim:[0,3,5],teacher_model:5,teacher_output:5,temperatur:3,tensor:1,test:5,text:[5,9],them:3,therefor:[3,9],thi:[0,3,9],three:10,tini:9,token:5,token_type_id:5,took:9,torch:[1,3,5,6,9],torchvis:6,torchvisiondatasetwrapp:9,tqdm:5,train:[1,3,4,5,6,9,12],train_it:5,trang:5,transfer:0,transform:[4,6,9,12],truncat:5,tupl:[3,5,6,9,11],twice:6,two:[3,9,10],type:[3,4,5,6,9,11],typeerror:0,union:[0,3,4,6,11],updat:3,use:[0,10],used:4,useful:0,using:[3,9],usual:0,util:[1,5,9],val_it:5,valid:[5,9,12],valid_metr:9,valu:[3,4],vision:[10,11],w_1:9,w_2:9,w_3:9,wai:[0,10],want:0,weight:[4,5,9],when:0,which:[0,6],wide:6,wide_resnet101_2:6,wide_resnet50_2:6,width_per_group:6,within:3,without:4,wrap:0,wrapper:[1,2],wrp:9,yet:4,you:[0,10],your:[0,10],zero_grad:5,zero_init_residu:6},titles:["Callbacks","Data","Distillation","Loss functions","Runners","Huggingface transformers","Computer Vision","Pruning","Quantization","Examples","Welcome to Compressors\u2019s documentation!","Models","&lt;no title&gt;"],titleterms:{"function":3,callback:0,complex:9,compressor:10,comput:6,data:1,differ:0,distil:2,document:10,exampl:9,hidden:0,huggingfac:5,logit:0,loss:3,minim:9,model:11,nlp:9,prune:7,quantiz:8,runner:4,state:0,transform:5,vision:6,welcom:10,why:10,wrapper:0}})