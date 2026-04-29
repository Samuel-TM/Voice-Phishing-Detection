# The University of Hong Kong

Department of Data and Systems Engineering M.Sc.(Eng.) in Robotics and Intelligent Systems 

DASE7099 Dissertation 

Interim Report 

Real-Time Dynamic Risk Tracking of Telecommunications Fraud Based on Parallel Multimodal Analysis 

SUN, Jiashan (3036523060) 

Supervisor: 

Dr. Shujia Qin 

# Abstract

With generative artificial intelligence (Generative AI) advancing rapidly, telecommunications (telecom) fraud is now a global public safety issue using deepfakes and highly tailored scripts instead of simple scripts the way it was previously done. According to leading industry research, telecom network fraud due to global markets is expected to have hundreds of billions of dollars in economic losses and the means of attack continue to have “industrialized” and “intelligent” behaviors. While significant growth has occurred in content-based fraud detection via research in academia, the majority of this research is concentrated on using a “post-hoc judgment” method of using full recordings or text of a phone conversation’s contents to determine if fraud occurred after the call has been completed. This mode cannot provide real-time intervention during the critical window of fraud, resulting in victims suffering economic losses before warnings are issued. This research aims to overcome this limitation by proposing a new paradigm: “in-event dynamic risk tracking.” 

The core methodology of this research is to construct a low-latency parallel multimodal streaming processing framework. This framework combines deep learning-based text semantic analysis with audio acoustic feature detection, and uses a sliding window mechanism to perform incremental risk assessment on real-time call streams, aiming to shift from simply “classification accuracy” to continuously tracking “risk evolution.” As of the mid-term stage in February 2026, this research has achieved the following key milestones: 

First, regarding the text analysis module, this study trained an enhanced robust Chinese-BERT model based on the TeleAntiFraud-28k dataset. During data preprocessing, in-depth log analysis revealed serious “conflict norms” and data duplication issues in the original data, 

which led to inflated validation set evaluations. To counter this issue, we utilized a series of methods for strict regularization, de-duplication and anti-leakage splitting strategies for datasets. More so, this study introduced the concept of “adversarial stress test” to confirm whether the model is actually able to comprehend semantics, i.e. whether or not it can actually ‘search’ for meanings in addition to memorizing a set of words as a result of the content in question. Experimental results show that the Chinese-BERT model maintains an F1 score above 0.99 even when facing adversarial examples with synonym replacement and sentence restructuring, while the traditional hard-rule-based keyword matching benchmark only achieves an F1 score of around 0.38, demonstrating the absolute advantage of deep learning models in semantic understanding. 

Secondly, in the audio analysis module, this study built and trained a CNN-BiLSTM acoustic detection model based on the ASVspoof 2019 LA dataset to identify synthetic speech. Regarding the severe model collapse phenomenon that occurred in the early training stage, which was caused by the large number of spoof samples compared to bonafide samples, resulting in the model’s predictions leaning towards a single category. This study adopted weighted random sampling and downsampling balance strategies. Additionally, to address the bottleneck of low feature extraction efficiency, a MFCC caching mechanism was introduced. The final model achieved an equal error rate (EER) of $1 7 . 9 8 \%$ and a balanced accuracy rate of 0.8150 on the validation set, effectively verifying its usability in low-latency environments. 

Currently, the research is in the transition stage from offline model validation to integration with real-time streaming frameworks. Although the single-modal components have achieved high performance, encapsulating them into an asynchronous streaming processing pipeline and implementing dynamic score fusion remain the main challenges in the next stage. 

# Declaration

I, SUN Jiashan, hereby declare that the M.Sc.(Eng.) Dissertation Interim Report, entitled "Real-Time Dynamic Risk Tracking of Telecommunications Fraud Based on Parallel Multimodal Analysis", which I am submitting, represents my own work and has not been previously submitted to this or any other institutions in the application for admission to a degree, a diploma or any other qualifications. 

Signed: SUN Jiashan 

Date: 22 Feb 2026 

# Table of Contents

# Declaration.

# Table of Contents . ii

# List of Figures . . iii

# List of Tables. . iv

# Chapter 1 Introduction

1.1 Research Background . . 

1.2 Problem Statement 

1.2.1 Research Problem Identification 

1.2.2 Research Objectives······ 2 

1.2.3 Research Issues and Tasks 2 

1.2.4 Research Scope 3 

1.2.5 Expected Outcomes 3 

1.3 Methodology Overview. . 3 

# Chapter 2 Literature Overview . 4

2.1 Paradigm Shift from Static Classification to Dynamic Tracking . . . 4 

2.2 Audio Deepfake Detection and the Evolution of ASVspoof 4 

2.3 Challenges of Real-Time Multimodal Streaming Processing . . 4 

# Chapter 3 Research Methodology and Initial Results 6

3.1 Dataset Preparation and Preprocessing . . . 6 

3.1.1 Text Dataset: TeleAntiFraud-28k 6 

3.1.2 Audio Dataset: ASVspoof 2019 LA 6 

3.2 Text Analysis Pipeline . . 6 

3.2.1 Model Architecture 6 

3.2.2 Challenges of Label Leakage and Spurious F1 7 

3.2.3 Verification of Adversarial Robustness 8 

3.3 Audio Analysis Pipeline . 9 

3.3.1 Model Architecture 9 

3.3.2 Challenges of Model Collapse ···· 9 

3.3.3 Solution and Final Result 10 

# Chapter 4 Progress Review and Roadmap 11

4.1 Progress Review . 11 

4.2 Major Updates . . 11 

4.3 Future Work Plan 

# Bibliography . . 13

# List of Figures

Figure 3.1 Architecture of the Chinese-BERT based text classification model · · · · 7 

Figure 3.2 Training curves and validation confusion matrix of Chinese-BERT Model 8 

Figure 3.3 Confusion matrix comparison between baseline and Chinese-BERT Model 9 

Figure 3.4 Architecture of the CNN-BiLSTM based synthetic speech detection model 9 

Figure 3.5 Dataset label distribution and attack algorithm statistics · · · · · · · · · · 10 

Figure 3.6 Training dynamics and performance evaluation of CNN-BiLSTM Model  10 

Figure 4.1 Gantt chart of the research schedule 12 

# List of Tables

Table 4.1 Project progress summary and milestone status · · · 11 

# Chapter 1 Introduction

# 1.1 Research Background

As the adoption of digital technologies for communication has taken off, and as the world continues to become more interconnected, telecommunications (telecom) fraud has grown from a problem related to social safety into a complex, asymmetrical war fueled by technology. In the current period of 2025-2026, the global telecom fraud situation is facing unprecedented and severe challenges. 

According to the “2025 Telecom Fraud Report” released by TransUnion, telecom fraud has become one of the most significant threats to global communication networks. Among the types of fraud that the surveyed companies are most concerned about, new user fraud and identity theft account for $53 \%$ and $50 \%$ respectively (TransUnion, 2025). Even more alarming, the latest data from Juniper Research shows that despite the continuous progress of anti-fraud technology, it is estimated that the global consumer losses caused by mobile messaging fraud will remain at a huge level of $\mathrm { U S S 7 1 }$ billion in 2026. Although this is a decrease from $\mathrm { U S } \$ 80$ billion in 2025, the absolute value is still shocking (Juniper Research, 2025). Especially in Hong Kong, with the active cross-border financial activities, fraudsters are using increasingly covert means of technology. TransUnion’s survey of the Hong Kong market shows that although the rate of suspected digital fraud attempts in Hong Kong has decreased in the first half of 2025, $26 \%$ of the surveyed companies still reported third-party fraud caused by identity theft, resulting in an estimated annual revenue loss of $\mathrm { H K S } 9 2$ billion, accounting for about $7 . 1 \%$ of the annual revenue of the surveyed companies (TransUnion Hong Kong, 2025). 

The most significant technological trend at present is the weaponization of generative artificial intelligence. The threshold for deepfake voice cloning technology has been greatly reduced, enabling fraudsters to generate realistic “familiar voices” or “authoritative voices” at very low cost. As proposed by Goodfellow et al. (2014), the principle of generative adversarial networks (GANs) has been maliciously abused to synthesize extremely realistic fake data. This technology allows fraud scripts to be dynamically adjusted according to the target’s real-time reaction, and traditional defense methods such as blacklisting are inadequate in the face of such highly dynamic attacks. Additionally, due to the growing sophistication of the fraud-as-a-service (FaaS) ecosystem, less experienced criminals can also develop sophisticated attacks (Gupta, 2025). 

# 1.2 Problem Statement

# 1.2.1 Research Problem Identification

Despite the large amount of money being invested by both academia and industry into anti-fraud technologies, current solutions have one primary flaw: poor timing. Almost all current fraud detection systems operate under a “post-event evaluation” model (Wickramasinghe et al., 2022). From metadata assessments using Call Detail Records (CDRs) (Balouchi et al., 2025) through current state-of-the-art deep machine learning techniques like transformer-based BERT (Devlin et al., 2019) and Wav2Vec models, the normal workflow of these systems involves creating a 

recording, making a phone call, analyzing the call, and providing feedback on the results. 

This method of recording, uploading, and processing calls has a significant lag time. In actual telecom fraud scenarios, especially those involving impersonation of public security, procuratorate and court officials, victims often complete the transfer in the last few minutes of the call. If the detection system can only make a judgment after the call ends, for the victims who have already been harmed, this judgment only has evidence value and completely loses its preventive value. 

In addition, existing single-modal detection methods are vulnerable to deepfake. Pure text analysis is prone to ignoring emotional abnormalities in speech, while pure audio analysis, although it can identify synthesized speech, cannot understand complex semantic inducement logic (Ali and Rajamani, 2012). Although multimodal fusion has become a research hotspot, current fusion research is mostly focused on offline architectures and lacks solutions adapted to low-latency streaming processing (Santos et al., 2025). 

# 1.2.2 Research Objectives

To address the lack of timeliness in existing technologies, this study establishes the following core objectives: 

1. A robust parallel multimodal detection framework was constructed and validated, which integrates a pre-trained, highly robust text and audio model that can resist both variant speech and voice forgery attacks (Ali and Rajamani, 2012). 

2. To address the engineering challenges of real-time streaming processing, a high-precision offline model is encapsulated in a low-latency asynchronous streaming pipeline, and incremental analysis of the call stream is achieved through a sliding window algorithm (Datar et al., 2002). 

3. It enables dynamic and visual tracking of risk evolution, and generates continuous risk probability curves through a dynamic risk scoring engine, providing users with intuitive real-time decision support. 

# 1.2.3 Research Issues and Tasks

To achieve the above goals, the core tasks of this study revolve around three major technical challenges. 

First, in terms of semantic robustness, the key task is to build a deep semantic model that can resist adversarial attacks such as synonym substitution and sentence restructuring (Ali and Rajamani, 2012), which has been initially verified by implementing an “adversarial stress test” on the TeleAntiFraud dataset (Ma et al., 2025). 

Second, in terms of acoustic anomaly detection, the core task is to solve the “model collapse” problem caused by extreme sample imbalance, which has been successfully solved by applying downsampling and weighting strategies on the ASVspoof dataset (Todisco et al., 2019). 

Finally, the core task of the next stage is streaming integration, which involves packaging the previously verified high-precision static model with low latency into an asynchronous streaming pipeline, and designing a dynamic scoring algorithm to convert the instantaneous predictions into smooth risk trends. 

# 1.2.4 Research Scope

The scope of this study is strictly limited to content risk control at the application layer, that is, identifying risks by analyzing the semantic and acoustic features of the call content. It does not involve the cracking of underlying communication protocols. At the data level, text analysis focuses on the common language context (based on TeleAntiFraud-28k), while audio analysis emphasizes general acoustic features (based on ASVspoof 2019 LA). The technical verification will be conducted in a simulated streaming environment rather than in the actual network deployment of telecom operators. 

# 1.2.5 Expected Outcomes

The expected outcomes of this research mainly include a set of highly robust detection model libraries that have been verified through adversarial testing (including the Chinese-BERT and CNN-BiLSTM models), a real-time tracking prototype system that integrates sliding windows and dynamic scoring algorithms, and a comprehensive evaluation report that includes timeliness analysis and adversarial test results. 

# 1.3 Methodology Overview

To solve these aforementioned issues, this paper proposes a “parallel multimodal streaming analysis framework.” Using a sliding window algorithm (Datar et al., 2002), this framework segments the audio from an extended calling session, e.g. a call made over VoIP, into fixed 30- second or 1-minute “snippets” that can then be uploaded as needed for subsequent processing and analysis. The technical roadmap for the framework includes the following: 

• Text Channel: Use a finely tuned Chinese-BERT model (Devlin et al., 2019) to perform semantic analysis on the transcribed text to capture fraudulent intent, such as inducing money transfer or asking for verification codes. 

• Audio Channel: The audio extracted from calls includes mel frequency cepstral coefficients (MFCC) features (Davis and Mermelstein, 1980) that can then be used in a CNN-BiLSTM architecture that combines a CNN and an LSTM for acoustic detection (Moussavou Boussougou and Park, 2023). The LSTM architecture has been widely used to analyze speech because it can effectively handle long-duration temporal relationships (Hochreiter and Schmidhuber, 1997). 

• Dynamic Risk Fusion: Using a dynamic risk-scoring engine to smooth the outputs of the text and audio channels provides a means to combine the risk values historically associated with windows of time into a single “risk vital sign map.” This provides a means to visually monitor fluctuations in the levels of risk associated with each call. In turn, using the criteria established in the technical roadmap, an alert will be generated to notify the appropriate person when the cumulative risk of the call exceeds the predetermined threshold. 

# Chapter 2 Literature Overview

# 2.1 Paradigm Shift from Static Classification to Dynamic Tracking

Early detection of telecom fraud relied primarily on graph mining of Call Detail Records (CDR), attempting to identify fraud syndicates through communication topology. However, with the popularization of VoIP and number spoofing technologies, metadata-based methods have gradually become ineffective (Balouchi et al., 2025). In recent years, the focus has shifted to contentbased detection. Ma et al. (2025) published TeleAntiFraud-28k. The dataset is a milestone in the field, providing high-quality audio-text pairs and introducing Slow-Thinking annotation logic. The release of the dataset has greatly promoted research on fraud detection based on semantic understanding. 

Shen et al. (2025) presented a paper at the CHI conference, which was the first to deeply explore the interactive design of real-time fraud detection based on large language models (LLMs), highlighting the trade-off between the timing of timeliness and accuracy. This is highly consistent with the concept of “dynamic risk tracking” proposed in this study. However, Shen’s research mainly focuses on the human-computer interaction level and only relies on the text modality, ignoring the importance of audio deep fake detection, which is a major gap in today’s rampant deepfake. 

# 2.2 Audio Deepfake Detection and the Evolution of ASVspoof

In the field of audio forensics, the ASVspoof challenge series has defined technical standards. ASVspoof 2019 first introduced the detection of TTS and VC attacks in logical access (LA) scenarios (Todisco et al., 2019). The subsequent ASVspoof 2021 further expanded to the detection of cross-channel and compressed audio (Liu et al., 2023). The latest ASVspoof 5 Challenge has begun to focus on adversarial attacks and field data detection in crowdsourced environments. Research by Wang et al. (2026) shows that traditional detectors are vulnerable to well-designed adversarial examples, highlighting the importance of model robustness. 

In real-time streaming scenarios, how to handle variable-length input, background noise, and silence segments remains an unresolved problem (Liu et al., 2023). In addition, the edge AI system proposed by Lu and Chen (2026) uses lightweight semantic voting to detect segmentbased voice fraud, demonstrating the feasibility of real-time detection on resource-constrained devices. 

# 2.3 Challenges of Real-Time Multimodal Streaming Processing

Multimodal learning has shown superior performance over single-modal learning in offline tasks (Santos et al., 2025). For example, the multimodal speech phishing detection system proposed by Ali and Rajamani (2012) integrates text and audio analysis, demonstrating that the fusion mechanism can effectively improve detection accuracy. However, applying multimodal models to streaming systems faces the contradiction of “high throughput and low latency”. Santos et al. (2025) pointed out that existing streaming systems are difficult to directly integrate 

large multimodal models (LMMs) because their inference speed often cannot keep up with the arrival speed of the data stream. To solve this problem, optimization must be carried out at the system architecture level, such as using model distillation, operator fusion and targeted streaming inference engines. In addition, research on adversarial attacks against AI models is also surging. Wang et al. (2025) showed that the Basic Iterative Method (BIM) attack against speech deepfake detectors can easily fool existing models. This further confirms the necessity of introducing “adversarial stress testing” into the text module in this study, as a single accuracy metric is no longer sufficient to measure the security of the system. 

# Chapter 3 Research Methodology and Initial Results

# 3.1 Dataset Preparation and Preprocessing

# 3.1.1 Text Dataset: TeleAntiFraud-28k

The TeleAntiFraud-28k dataset, as described by Ma et al. (2025), comprises about 28,000 expertly labeled texts representing both fraudulent and legitimate phone calls. Furthermore, this dataset contains various scripts and scenarios used for perpetrating scams. 

During the initial exploration of the TeleAntiFraud-28k dataset for the purpose of creating an active text classification model, we observed major quality issues with the dataset. These quality issues included a multitude of what are referred to as “conflict norms,” or instances in which the same text was labeled with opposing class labels. To remedy this issue, we put into effect a highly structured normalizing deduplication process. All textual data were normalized first; then, all duplicate labeled records were identified and excluded. Subsequently, the original global record set was deduplicated prior to splitting into the training and validation sets. This deduplication reduced the number of samples from 16070 to 16061. 

# 3.1.2 Audio Dataset: ASVspoof 2019 LA

ASVspoof 2019 Logical Access (LA) Dataset (Todisco et al., 2019) served as the foundation for the audio classifier’s training. This dataset is an authoritative standard for assessing the accuracy of automated deep fake audio detection systems that utilize real (bonafide) and attack (spoofed) audio samples produced by many forms of automatic speech generation (TTS or VC) algorithms. 

To accommodate lightweight models and meet real-time requirements, we employed mel frequency cepstral coefficients (MFCC) features (Davis and Mermelstein, 1980). 13-dimensional MFCC features (n_mfcc=13, n_ff ${ = } 2 0 4 8$ , hop_length $= 5 1 2$ ) were extracted from the audio signal using the librosa library (McFee et al., 2015), and subject to a maximum limit of 500 frames per sample. These MFCC features can effectively replicate the hearing properties found in human ears, and at the same time they significantly compress the amount of data that must be stored while maintaining the most important components of the speech signal. 

To expedite the time-consuming MFCC extraction process, a local caching mechanism has been developed for training the models (MFCC Cache). The features extracted from the audio samples will be saved to .npy files, which allows for the subsequent re-using of these features in future epochs of model training, thereby shortening the overall amount of time required for training the models while improving the effectiveness of the experiments. 

# 3.2 Text Analysis Pipeline

# 3.2.1 Model Architecture

To implement the text module, we used Chinese-BERT (Devlin et al., 2019). Unlike the regular Term Frequency-Inverse Document Frequency (TF-IDF) that can capture only one direction of 

context, BERT is able to capture both bidirectional, context of the words. This is important in understanding both the complex “scripts for fraud” and the many contextually dependent parts of those scripts. As shown in Figure 3.1, the model architecture consists of a BERT encoder, dropout layer to reduce the likelihood of overfitting (rate $= 0 . 3$ ) and a fully connected layer to produce probabilities for two classes. The model was implemented using PyTorch (Paszke et al., 2019) and evaluated using scikit-learn (Pedregosa et al., 2011). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/7d62b1b090c9aa06a0a0cb92773f2f7f3aacb37036fa9e622306c92c90b92107.jpg)



Figure 3.1 Architecture of the Chinese-BERT based text classification model


# 3.2.2 Challenges of Label Leakage and Spurious F1

As detailed in Figure 3.2, an unexpected finding occurred in early rounds of experimentation when the validation set achieved an F1 score of 1.0 within just a few iterations. Upon further analysis, we learned that this was not due to the excellence of the model’s performance; rather, there were issues with “label leakage.” Some of the label keywords (like “fraud” and “normal”) were present in the original input JSONL files as response fields; then, in some instances, these response fields were concatenated in the input text during preprocessing, which gave the model an advantage in learning these label keywords instead of learning about the content of the inputted text. Thus, in restructuring “TeleAntiFraudDataLoader,” we made certain that no metadata fields with label information were included so that the model only evaluated inputs based upon call semantics. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/d9c2ce57cb26fbd50899335cae27f757a4e18eb401b5350e0629d68dc489ea1f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/cbbdf71427d52f276ac348a21f689acc85df58c5f077bf586a29b5568f59722d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/a75f0a9f5c4880434adae4b7be64a16a5ec8da78ebe8254403caa627f15d2c6d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/8bff19730105fb9096839993815c70ad980ac568d208db964356ce9342c68d79.jpg)



Figure 3.2 Training curves and validation confusion matrix of Chinese-BERT Model


# 3.2.3 Verification of Adversarial Robustness

To validate that our use of Chinese-BERT had truly learned semantic structure instead of merely memorised words, we created an adversarial stress test to examine whether or not it produced results that were contrary to expectation. We created a benchmark of “hard keyword rules” using regular expressions and used that as a reference model to compare against the outputs of our model on samples of adversarial frauds (adv_phish) that have been subjected to synonym substitutions and restructuring of their sentences. The results of the comparison with traditional keyword benchmarks show a stark contrast: the performance of traditional benchmarks drops sharply, with an F1 score of only 0.3857; while Chinese-BERT demonstrates excellent robustness, with an F1 score as high as 0.9962 (Figure 3.3). This result strongly demonstrates that our model can understand the deep semantic structure of fraudulent rhetoric, giving it strong resistance to interference when facing constantly changing attack scripts. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/33d8a869dc30714f77438796510cac6182034f6088949ac9bc9cb7454417c904.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/fbc17c1ed91387eefed70ebe594b2cdfea6724ceaacafc55a38bdd31f25ab9c5.jpg)



$= = =$ Summary (Leak-safe Adversarial Test) $= = =$ HardKeyword $\mathsf { F } 1 = \mathsf { \boldsymbol { \theta } } . 3 8 5 7$ ，AUC=0.5044 | Chinese-BERT F1=0.9962，AUC=1.0000



Figure 3.3 Confusion matrix comparison between baseline and Chinese-BERT Model


# 3.3 Audio Analysis Pipeline

# 3.3.1 Model Architecture

The audio module uses CNN-BiLSTM hybrid architecture (Moussavou Boussougou and Park, 2023). The front end consists of a one-dimensional convolutional layer (Conv1d) to capture local frequency domain features and texture information of MFCC; the back end is connected to a bidirectional long short-term memory network (BiLSTM) (Hochreiter and Schmidhuber, 1997) to capture long temporal dependencies of speech. As shown in Figure 3.4, this architecture can take into account both local features and global context when processing speech sequence tasks, and is very suitable for recognizing unnatural prosody and artifacts in synthesized speech. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/38958f52321388bd0bd2b453a8bbe2725bbdc49c04d0edd7ebbdd237664a6f00.jpg)



Figure 3.4 Architecture of the CNN-BiLSTM based synthetic speech detection model


# 3.3.2 Challenges of Model Collapse

In the early stages of training, we encountered a severe model collapse problem. As illustrated in Figure 3.5, the ASVspoof training set had far more spoof samples (22,800) than bonafide 

samples (2,580), a ratio close to 9:1. This extreme class imbalance caused the model to tend to predict all inputs as the majority “spoof” class. While this resulted in seemingly high overall accuracy, the model effectively lost the ability to recognize human speech. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/b19135af56e7ca18c37e5f499a9a374da52feb3cb7e3d2b88a5adef09e3cd21c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/fdf28a9c54deef8f89689474edd230702227c12ee859f5315ef95f317b77a954.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/16ca93d89ebd035b5a5e1cc41ece38bb235110cc4acb2c6d0725a6aeb71f56fb.jpg)



Figure 3.5 Dataset label distribution and attack algorithm statistics


# 3.3.3 Solution and Final Result

To address the model collapse problem caused by extreme sample imbalance, we adopted a strategy combining downsampling and weighted random sampling to successfully construct a 1:1 balanced training set. After 14 epochs of training, this strategy achieved remarkable results. Ultimately, the audio detection model achieved an equal error rate (EER) equal to $1 7 . 9 8 \%$ and a balanced accuracy of 0.8150 for the validation set; the F1 score for fraudulent speech (spoof) was 0.9004; and the recall rate for real speech (bonafide) was approximately $80 \%$ (Figure 3.6). These indicators collectively verified the reliability of the model in effectively distinguishing between true and false voices, providing valuable acoustic judgment basis for multimodal systems (Yi et al., 2023). 


CNN-BiLSTM (ASVspoof 2019 LA)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/de254a59e48a4d42e381d38a6384f939dbbdd5e0e32f163666c071e17484e675.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/5010d8017c1942ff8ef0ce2e7d3b8b3ad5bd224f672d18e7b513ab5e77880c63.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/0b999e5bbec69d59ff8c2a277a9212932d7c62018cf9949594111ffc6d6bbbe3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/c2ec147a0062f8ec006e1b4ed37559c3b5538d28b171ca714730880ba69be38a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/82b987b30de45eaad80fe9d6f181117c99041c4f5609b15c8a1ba8267b788ba6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/6a65c90ee88631ca297ba25c2e4d005f558ff972f21e78bf3f7f106dc2241d4a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/1fecad28d8395d287ccfb047afd00a19ddc15549960bd3249a65829f1971b5f9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/53d3d57a32dddd5869616228c1a5c85ddaa4258623ed18a9741daec5bf93b4a9.jpg)



Figure 3.6 Training dynamics and performance evaluation of CNN-BiLSTM Model


# Chapter 4 Progress Review and Roadmap

# 4.1 Progress Review

The following table summarizes the current status of the research modules as of February 2026: 

<table><tr><td>Task module</td><td>Sub-tasks</td><td>Completeness (%)</td><td>Status and Gap</td></tr><tr><td>1. Offline model building</td><td>Text model</td><td>100%</td><td>Training completed. 
Strong robustness achieved (F1&gt;0.99).</td></tr><tr><td></td><td>Audio model</td><td>100%</td><td>Training completed. 
Model collapsed 
resolved (EER ~17.98%).</td></tr><tr><td>2. Robustness verification</td><td>Adversarial testing</td><td>100%</td><td>Adversarial testing of the text modality has been completed.</td></tr><tr><td>3. Streaming framework</td><td>Sliding window</td><td>10%</td><td>Design phase only; coding pending.</td></tr><tr><td>4. Dynamic scoring engine</td><td>Risk fusion logic</td><td>20%</td><td>Logic defined; verification pending.</td></tr></table>


Table 4.1 Project progress summary and milestone status


# 4.2 Major Updates

Compared to the initial research proposal, this research made key adjustments based on experimental feedback: The audio analysis shifted from the original waveform to more efficient MFCC features to meet the real-time requirements; the text analysis additionally introduced adversarial evaluation to verify the deep semantic understanding ability of the model; and a more aggressive data balancing strategy was adopted to solve the model collapse problem. 

# 4.3 Future Work Plan

The focus of the following work will be entirely shifted to system integration. The detailed schedule is visualized in the Gantt chart below. 

• Phase 1: Streaming Pipeline Development (March 2026) 

• Phase 2: Dynamic Risk Scoring Engine (April 2026) 

• Phase 3: System Evaluation and Metrics (May 2026) 

• Phase 4: Thesis Writing and Defense (June 2026 - August 2026) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/29872ac4-9cd7-44c8-9476-527913be6087/e5a600d1b1d904de133955aaf14bbd5c041a6a2cee9a41feda20730271c453ef.jpg)



Figure 4.1 Gantt chart of the research schedule


# Bibliography



Ali, M. M. & Rajamani, L. (2012). Deceptive phishing detection system: from audio and text messages in instant messengers using data mining approach. In International Conference on Pattern Recognition, Informatics and Medical Engineering (PRIME-2012), pages 458–465. IEEE. 





Balouchi, A., Abdollahi, M., Eskandarian, A., et al. (2025). Wangiri fraud detection: A comprehensive approach to unlabeled telecom data. Future Internet, 18(1), 15. 





Datar, M., Gionis, A., Indyk, P., et al. (2002). Maintaining stream statistics over sliding windows. SIAM Journal on Computing, 31(6), 1794–1813. 





Davis, S. & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357–366. 





Devlin, J., Chang, M. W., Lee, K., et al. (2019). Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics, pages 4171–4186. 





Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., et al. (2014). Generative adversarial nets. In Advances in Neural Information Processing Systems, volume 27. 





Gupta, R. (2025). National cyber threat assessment 2025-2026. Communications Security Establishment Canada. 





Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780. 





Juniper Research (2025). Mobile messaging fraud prevention market 2025-2030. Juniper Research. Accessed: February 5, 2026. 





Liu, X., Wang, X., Sahidullah, M., et al. (2023). Asvspoof 2021: Towards spoofed and deepfake speech detection in the wild. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31, 2507–2522. 





Lu, S. Y. & Chen, W. P. (2026). Edge ai system using lightweight semantic voting to detect segment-based voice scams. Engineering Proceedings, 120(1), 14. 





Ma, Z., Wang, P., Huang, M., et al. (2025). Teleantifraud-28k: An audio-text slow-thinking dataset for telecom fraud detection. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 5853–5862. 





McFee, B., Raffel, C., Liang, D., et al. (2015). librosa: Audio and music signal analysis in python. SciPy. 





Moussavou Boussougou, M. K. & Park, D. J. (2023). Attention-based 1d cnn-bilstm hybrid model enhanced with fasttext word embedding for korean voice phishing detection. Mathematics, 11(14), 3217. 





Paszke, A., Gross, S., Massa, F., et al. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32. 





Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in python. Journal of Machine Learning Research, 12, 2825–2830. 





Santos, U. J. L., Ferri, A., Nistor, S., et al. (2025). Towards a multimodal stream processing system. arXiv preprint arXiv:2510.14631. 





Shen, Z., Yan, S., Zhang, Y., et al. (2025). "it warned me just at the right moment": Exploring llm-based real-time detection of phone scams. In Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, pages 1–7. 





Todisco, M., Wang, X., Vestman, V., et al. (2019). Asvspoof 2019: Future horizons in spoofed and fake audio detection. arXiv preprint arXiv:1904.05441. 





TransUnion (2025). 2025 telecom industry fraud report. TransUnion. Accessed: February 5, 2026. 





TransUnion Hong Kong (2025). 2025 h2 top fraud trends report. TransUnion Hong Kong. Accessed: February 5, 2026. 





Wang, W. E., Salvi, D., Negroni, V., et al. (2025). Bim-based adversarial attacks against speech deepfake detectors. Electronics, 14(15), 2967. 





Wang, X., Delgado, H., Evans, N., et al. (2026). Asvspoof 5: Evaluation of spoofing, deepfake, and adversarial attack detection using crowdsourced speech. arXiv preprint arXiv:2601.03944. 





Wickramasinghe, Y., Rathnayake, P., Madumal, O., et al. (2022). A review on telecommunication frauds and fraud detection techniques. arXiv preprint. 





Yi, J., Tao, J., Fu, R., et al. (2023). Add 2023: the second audio deepfake detection challenge. arXiv preprint arXiv:2305.13774. 

