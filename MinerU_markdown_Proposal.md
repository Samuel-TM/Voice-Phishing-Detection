# The University of Hong Kong

Department of Data and Systems Engineering M.Sc.(Eng.) in Robotics and Intelligent Systems 

DASE7099 Dissertation 

Proposal 

Real-Time Dynamic Risk Tracking of Telecommunications Fraud Based on Parallel Multimodal Analysis 

SUN, Jiashan (3036523060) 

Supervisor: 

Dr. Shujia Qin 

# Abstract

Telecommunications (Telecom) fraud has become an increasingly severe global social problem. Although many advancements have been made in detection technology in both academia and industry, existing research is generally limited to one core paradigm: "Post-hoc Judgment" of complete call recordings or text records. No matter how high the classification accuracy of this model is, it cannot provide users with immediate protection during the occurrence of fraud, and thus it is difficult to effectively prevent economic losses. This research aims to break through this limitation and proposes and explores a brand-new "In-event Dynamic Risk Tracking" paradigm for the first time. The core objective of the research is to shift from pursuing the "accuracy of final classification" to the "timeliness of early warning". By conducting realtime and continuous assessment and visualization of risk evolution during calls, it provides users with decision support to terminate calls in the early stage of fraud. 

To achieve this goal, this study has set four major research objectives: Firstly, systematically review and critically analyze the existing fraud detection research, demonstrate the widespread limitations of "post-event analysis", and thereby establish the necessity of the "inprocess tracking" paradigm; Secondly, design and develop a low-latency, multimodal parallel processing framework that can support real-time stream analysis; Secondly, verify the feasibility of training dedicated analysis models using existing large-scale public datasets and integrating them into a new framework; Finally, by introducing time-sensitive evaluation indicators, the early warning timeliness and dynamic risk tracking capability of the developed system are comprehensively evaluated. 

This study intends to adopt an innovative multimodal parallel real-time detection architec-

ture. This method first preprocesses the real-time audio stream through speaker separation technology to cope with multi-party call scenarios. Subsequently, the system processes the audio clips of each speaker through a sliding window mechanism and distributes them to two parallel analysis pipelines: a text analysis pipeline that uses Automatic Speech Recognition (ASR) and lightweight Large Language Models (LLM) for deep semantic analysis; And an audio analysis pipeline that uses acoustic features and deep learning models to identify abnormal patterns of sound. The analysis results of the two pipelines will be fused through a dynamic risk scoring engine to generate and continuously update a fraud risk probability curve. 

The expected outcome of this research includes a rigorously validated system prototype embodying the "in-event tracking" concept. This study aims to promote the evolution of telecom fraud detection from static, post-hoc classification tasks to dynamic, real-time risk tracking tasks, exploring a more practically valuable technological direction. 

# Declaration

I, SUN Jiashan, hereby declare that the M.Sc.(Eng.) Dissertation proposal, entitled "Real-Time Dynamic Risk Tracking of Telecommunications Fraud Based on Parallel Multimodal Analysis", which I am submitting, represents my own work and has not been previously submitted to this or any other institutions in the application for admission to a degree, a diploma or any other qualifications. 

Signed: SUN Jiashan 

Date: 9 Nov 2025 

# Table of Contents

# Declaration.

# Table of Contents . ii

# List of Figures . . iii

# List of Tables. . iv

# Chapter 1 Introduction 1

1.1 Research Background . 

1.2 Detailed Research Questions. . 

1.2.1 Research Problem Identification 

1.2.2 Research Objectives···· 2 

1.3 Overview of Research Methods . 2 

# Chapter 2 Literature Overview 4

2.1 The Post-Hoc Judgment Paradigm and Its Limitations. . . 4 

2.2 Exploration and Limitations of Real-time Detection Methods . 5 

2.3 Status and Challenges of Multimodal Fusion . . 5 

2.4 Research Gap in the Dynamic Tracking Paradigm . . . 5 

# Chapter 3 Proposed Research Methods 6

3.1 System Architecture. . 6 

3.2 Dataset Strategy . . . 6 

3.3 Text Analysis Pipeline . . 7 

3.4 Audio Analysis Pipeline . . 7 

3.5 Dynamic Risk Scoring and Decision Logic . . . 7 

# Chapter 4 Research Plan and Milestones . 9

# Bibliography . . 10

# List of Figures

Figure 4.1 Gantt Chart of the Research Schedule · · · · · 9 

# List of Tables

# Chapter 1 Introduction

# 1.1 Research Background

Telecom fraud, as a constantly evolving form of crime, poses a serious threat to personal property safety and social stability worldwide. Recent research highlights that with the popularization of Generative AI technology, fraud methods have become increasingly complex (Figueiredo et al., 2024), leading to rising economic losses year by year (Boucha and Ahmad, 2022). Faced with such a severe situation, developing effective technological prevention methods has become a common focus of academia and industry. 

However, a fundamental problem is that the vast majority of current telecom fraud detection research follows a common research paradigm: "Post-hoc Judgment." This applies to both traditional analysis based on Call Detail Records (CDR) (Wickramasinghe et al., 2022) and modern content analysis methods, which typically involve inputting a complete call recording or text transcript into a classification model to output a binary "fraud" or "non-fraud" label (Bhat, 2021). The core research objective under this paradigm is to continuously improve the accuracy, precision, and recall of the classification model. 

Despite significant achievements in model accuracy in these studies, they collectively overlook a crucial practical application requirement: timeliness. For users caught in a scam, a detection system that only provides results after the call has ended, no matter how high its accuracy, loses its practical protective meaning, as economic losses often occur within the last few minutes of the call. Therefore, the existing "post-hoc judgment" research paradigm has an inherent and insurmountable gap in protecting users from immediate harm. 

# 1.2 Detailed Research Questions

# 1.2.1 Research Problem Identification

This core problem marks a significant shift in research focus, no longer just on "is this call a scam?" but on "how is the risk of this call evolving?" This leads to several key technical and theoretical challenges, including the challenge of paradigm shift, the challenge of real-time and latency, the challenge of incomplete information, and the challenge of evaluation methods. First, how to reconstruct a static binary classification problem into a dynamic time series analysis problem, which requires us not only to judge but also to track and predict the continuous changes in risk. Second, "in-event" analysis requires the system to complete the processing, analysis, and decision-making of continuous audio streams with extremely low latency; any significant delay will degrade "in-event" to "post-hoc." Furthermore, in the early stages of a call, due to very limited information available for analysis, how to make preliminary but reliable risk assessments with insufficient information and dynamically correct judgments as information increases is key to achieving early warning. Finally, traditional metrics such as accuracy cannot measure "timeliness," requiring the exploration of new evaluation metrics and methods to quantify the early warning capability of a dynamic tracking system. 

# 1.2.2 Research Objectives

To address the above research problems, this study sets the following four main objectives: 

1. Objective 1: To conduct in-depth research and systematically review the current technical status of telecom fraud detection, and through critical analysis, demonstrate the widespread "post-hoc analysis" limitations of existing research, thereby establishing the theoretical necessity and innovative value of the new "in-event dynamic risk tracking" paradigm proposed in this study. 

2. Objective 2: To design and develop a novel, low-latency multimodal analysis framework that supports real-time streaming processing. This framework will be capable of short-term incremental analysis of call audio streams and continuously outputting a dynamically changing risk score, forming a risk evolution curve. 

3. Objective 3: To verify the feasibility of training specialized analysis models using existing large-scale public datasets, e.g. TeleAntiFraud and ASVspoof, and integrating them into the proposed dynamic tracking framework. 

4. Objective 4: To evaluate the performance of the developed system prototype, not only assessing its final classification accuracy but also focusing on its early warning timeliness, e.g. the ability to issue alerts before critical fraudulent instructions appear, and effectiveness of risk tracking. 

# 1.3 Overview of Research Methods

To achieve the above research objectives, this study proposes to adopt a real-time dynamic risk tracking method based on parallel multimodal analysis. The core idea of this method is to transform traditional, one-time classification judgments into a continuous, incremental risk assessment process. 

First, the system will use Speaker Diarization technology to preprocess the input audio stream. This step aims to segment complex dialogue streams in real-time into independent audio segments belonging to different speakers, e.g. fraudsters and victims, laying the foundation for subsequent precise analysis. 

Subsequently, the separated audio segments will enter a Sliding Window based streaming processing framework (Datar et al., 2002). The system will analyze audio data from the most recent 10-second window at fixed time intervals. Data from each window will be distributed to two parallel analysis pipelines. 

The text analysis pipeline uses efficient Automatic Speech Recognition (ASR) technology to transcribe audio into text in real-time. Subsequently, a lightweight Large Language Model (LLM) optimized for real-time applications will perform deep semantic analysis of the transcribed content to identify fraudulent language and intent. 

The audio analysis pipeline directly processes raw audio signals, extracting acoustic features such as Mel-frequency Cepstral Coefficients (MFCCs) and using a pre-trained deep learning model, e.g. a CNN-BiLSTM architecture (Moussavou Boussougou and Park, 2023), to detect acoustic artifacts specific to AI-synthesized speech or abnormal emotional prosody. 

Most critically, the outputs of the two pipelines are no longer final classification labels, but probability scores representing the degree of risk within the current window. These scores will be fed into a Dynamic Risk Scoring Engine. This engine will combine historical scores, using weighted or time-series models to calculate a comprehensive, smoothed current risk value. This process is continuously repeated during the call, thereby generating a real-time, dynamically changing risk probability curve, providing users and the system with intuitive insights into risk evolution trends. 

# Chapter 2 Literature Overview

This chapter aims to demonstrate the pervasive "post-hoc judgment" limitations of current telecom fraud detection technologies through a systematic review and critical analysis of the research status in related fields, thereby highlighting the necessity and innovativeness of the new "in-event dynamic risk tracking" paradigm proposed in this study. 

# 2.1 The Post-Hoc Judgment Paradigm and Its Limitations

Early telecommunications fraud detection technologies mainly relied on the analysis of metadata in communication networks. Researchers analyzed Call Detail Records (CDRs) and used machine learning algorithms to identify abnormal call patterns. In their systematic review, Wickramasinghe et al. (2022) pointed out that these methods have certain effects in identifying largescale and patterned fraud activities. However, the fundamental limitation of such technologies lies in their inability to touch upon the actual content of the call. As fraud methods become increasingly socially engineered and personalized, it has become difficult to effectively identify carefully disguised fraud calls merely by analyzing metadata. Secondly, the essence of these methods is to conduct batch processing and offline analysis of completed call data, which is a typical "post-event judgment" (Wickramasinghe et al., 2022). Its main purpose is to provide a basis for operators to identify fraudulent numbers, rather than to protect users during calls. 

With the development of technology, the research focus shifted to direct analysis of call content. This stage is mainly divided into two parallel technical routes: text-based analysis and audio-based analysis. 

In the field of text analysis, researchers use Automatic Speech Recognition (ASR) technology to transcribe call recordings into text, and then apply Natural Language Processing (NLP) models for classification. In recent years, with the rise of Large Language Models (LLM), this direction has made significant progress, as studies like Shen et al. (2025) demonstrate the feasibility of using LLMs to analyze dialogue semantics for real-time detection. However, these methods also have obvious shortcomings. First, they rely almost entirely on transcribed text information, ignoring the audio signal itself. When AI voice cloning technology is advanced enough that the generated voice sounds indistinguishable from a real person, its transcribed text may be exactly the same as the transcribed text of real speech, rendering pure text analysis methods ineffective. Second, the evaluation methods of these studies are almost without exception based on complete call text transcripts. 

In the audio analysis field, the core of the research is Audio Deepfake Detection, which involves finding traces of forgery from acoustic features (Zhang et al., 2025). Research in this field has greatly promoted the identification technology of AI-synthesized speech, especially models based on hybrid architectures such as CNN-BiLSTM (Moussavou Boussougou and Park, 2023). However, the research paradigm in this field is also "post-hoc judgment." Its authoritative benchmark evaluations, such as the ASVspoof series challenges, are set to perform binary classification of true and false for given, complete, usually several-second audio segments (Wang et al., 2020). These evaluations do not consider scenarios for real-time analy-

sis in continuous, uninterrupted call streams. 

Whether based on text or audio, these mainstream studies share a commonality: they treat fraud detection as a standard, static classification problem, where the input is a complete, clearly bounded data sample (a complete recording or text) and the output is a final, definite label. 

# 2.2 Exploration and Limitations of Real-time Detection Methods

In recent years, a few forward-looking studies have begun to realize the limitations of "post-hoc judgment" and have taken the first step towards "real-time detection." Among them, the most representative is the LLM-based real-time detection framework proposed by Shen et al. (2025). Their system can, during a call, in a "turn-by-turn" manner, make a judgment on the current dialogue history after each speaker finishes speaking, and issue an alert when fraudulent intent is detected. 

This work is undoubtedly an important step towards practical application, but its underlying research paradigm has not completely escaped the shackles of "post-hoc judgment." Specifically, the task of the system at each time point is still a binary classification (or ternary, if an "uncertain" state is included), and its goal is to make a "correct" classification decision as quickly as possible at the current time point. This method can be regarded as a "rolling, shortterm ’post-hoc judgment’." Although it improves response speed, it still has several fundamental problems: fragility of decision-making, lack of a risk evolution perspective, and limitations of evaluation metrics. 

# 2.3 Status and Challenges of Multimodal Fusion

To overcome the limitations of unimodal methods, multimodal detection methods have emerged. For instance, research by Kim et al. (2025) has shown that combining textual semantic information and acoustic features of audio can achieve better classification performance than any single modality. However, current multimodal research is also mostly limited to offline analysis frameworks (Tan and et al., 2025) and faces huge challenges in real-time fusion, such as ensuring low-latency inference and managing the synchronization of different modal data streams (Weller et al., 2021). 

# 2.4 Research Gap in the Dynamic Tracking Paradigm

Through a systematic review of the above literature, this study identifies a profound and pervasive research gap: the entire field of telecom fraud detection, whether unimodal or multimodal, currently lacks a systematic "dynamic risk tracking" research paradigm with "in-event early warning" as its core objective. Existing research defines the problem as one or a series of independent classification tasks, while this study advocates redefining it as a time series risk estimation and tracking task. 

# Chapter 3 Proposed Research Methods

This chapter will elaborate on the overall technical solution designed to achieve the core objective of "in-event dynamic risk tracking," focusing on the functional positioning and selection basis of each core module. 

# 3.1 System Architecture

The system proposed in this study adopts a multimodal parallel processing architecture, whose design fully serves the real-time, low-latency dynamic risk tracking objective. Its core design ideas include: 

• Speaker Diarization Preprocessing: As the first step of the system, all input audio streams will pass through a speaker separation module. The core task of this module is to segment mixed dialogue streams in real-time into independent audio segments belonging to different speakers, e.g. fraudsters and victims. This step is crucial for accurate, individual-specific risk assessment in multi-party calls. 

• Streaming Processing and Parallel Analysis: After speaker diarization, the system will use a Sliding Window mechanism to stream process audio segments of each speaker (Datar et al., 2002). Data within each time window will be distributed to two parallel analysis pipelines. This parallel design ensures that time-consuming computing tasks can be carried out simultaneously, thereby minimizing end-to-end latency to the greatest extent (Weller et al., 2021). 

• Dynamic Risk Scoring and Update: The outputs of the two parallel pipelines are probability scores representing the current risk level, rather than the final decision. A Dynamic Risk Scoring Engine will fuse these immediate scores with historical scores based on the preset weighting strategy and time series smoothing algorithm, calculate and update a comprehensive and continuously changing risk probability. 

• Real-time Visualization Frontend: To visually present the test results, this study will develop a simple visualization interface. This interface will present in real time the evolution trend of fraud risk probability with call duration in the form of a dynamic curve graph, providing users with intuitive decision support. 

# 3.2 Dataset Strategy

This study will fully utilize two existing, high-quality public datasets to train and evaluate different analysis modules of the system, respectively, to verify the effectiveness of the method. 

First, for the text analysis dataset, this study will use TeleAntiFraud- ${ } ^ { 2 8 \mathrm { k } }$ , an open-source speech-text dataset specifically designed for telecom fraud analysis (Ma et al., 2025). It provides a large number of text samples with fraud reasoning annotations, which are very suitable for training and evaluating the fraud intent recognition model in the text analysis pipeline. 

Second, for the audio analysis dataset, this research will leverage ASVspoof, which is an authoritative benchmark dataset series in the field of audio deepfake detection (Wang et al., 2020). It contains a large number of real speech and deceptive audio files generated by various advanced algorithms, making it an ideal choice for training and evaluating the AI-synthesized speech detection model in the audio analysis pipeline. 

# 3.3 Text Analysis Pipeline

This module is responsible for real-time mining of fraud intentions from the dialogue content. Its workflow is conceptually divided into two steps. Firstly, a high-performance real-time Automatic Speech Recognition (ASR) model that supports streaming inference will transcribe the input audio clips into text in real time. Selecting a low-latency and high-accuracy ASR model (such as a streaming variant based on Whisper) is the key to ensuring the real-time performance of the entire system. Subsequently, a lightweight Large Language Model (LLM) optimized for real-time applications will analyze the transcribed text. Unlike traditional methods, this model not only analyzes the current text but also combines the limited historical dialogue context to output a probability score representing the semantic risk at the current moment. 

# 3.4 Audio Analysis Pipeline

This module focuses on rapidly identifying acoustic anomalies in the audio signal itself to determine whether it is AI-synthesized or exhibits abnormal emotions. First, the acoustic feature extraction step is responsible for extracting features that can effectively characterize speech properties from the raw audio waveform. MFCCs will be used as the main acoustic features in this study because they effectively simulate human ear auditory characteristics and have been widely proven to be very effective in distinguishing real from synthetic speech. Subsequently, acoustic anomaly detection will be performed using a deep learning model to analyze the extracted acoustic feature sequences. Considering the need to capture both local artifacts and longterm temporal dependencies in audio, this study plans to adopt a hybrid architecture combining Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory Networks (BiLSTM), an approach similar to that used by Moussavou Boussougou and Park (2023). This model will be trained on datasets such as ASVspoof to learn to distinguish acoustic patterns of real speech and various AI-synthesized speech, and output a probability score representing the acoustic risk at the current moment. 

# 3.5 Dynamic Risk Scoring and Decision Logic

This module is the core of realizing the "in-event tracking" paradigm. It is responsible for transforming the discrete risk scores from the two parallel pipelines into a continuous, smooth risk curve. This study will explore a method based on weighted moving average or more complex time-series filtering algorithms to combine the semantic risk probability output by the text pipeline and the acoustic risk probability output by the audio pipeline into a comprehensive risk score. This fusion mechanism can achieve cross-modal collaborative verification and smooth out misjudgment jitters caused by insufficient information in a single window, a challenge also noted by Shen et al. (2025), thereby reflecting the overall risk evolution trend more stably and reliably. 

# Chapter 4 Research Plan and Milestones

This chapter outlines the research plan, dividing the project into four distinct phases. Each phase has specific objectives and a clear timeline, culminating in the final thesis submission in August 2026. The detailed schedule is visualized in the Gantt chart below. 

• Phase 1: Foundation Preparation (September 2025 - November 2025) 

• Phase 2: Data and Module Development (December 2025 - February 2026) 

• Phase 3: System Integration and Testing (March 2026 - May 2026) 

• Phase 4: Thesis Writing and Defense (June 2026 - August 2026) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-29/111e2591-9ac7-4ef4-b3e2-5e57e4eb1c31/6c33e46ed9a06399a3f1b46ceb47640be6e86e40314abb9cc9ec2f12007e7e6a.jpg)



Figure 4.1 Gantt Chart of the Research Schedule


# Bibliography



Bhat, S. M. (2021). A review of telecommunication fraud detection techniques. In Proceedings of the International Conference on Innovative Computing & Communication (ICICC). 





Boucha, A. & Ahmad, R. (2022). A literature review of financial losses statistics for cyber security and future trend. World Journal of Advanced Research and Reviews, 15(1), 138–156. 





Datar, M., Gionis, A., Indyk, P., & Motwani, R. (2002). Maintaining stream statistics over sliding windows. SIAM Journal on Computing, 31(6), 1794–1813. 





Figueiredo, J., Carvalho, A., Castro, D., Gonçalves, D., & Santos, N. (2024). On the feasibility of fully ai-automated vishing attacks. arXiv preprint arXiv:2409.13793. 





Kim, J., Gu, S., Kim, Y., Lee, S., & Kang, C. (2025). A multimodal voice phishing detection system integrating text and audio analysis. Applied Sciences, 15(20), 11170. 





Ma, Z., Wang, P., Huang, M., Wang, J., Wu, K., Lv, X., ..., & Kang, Y. (2025). Teleantifraud-28k: An audio-text slow-thinking dataset for telecom fraud detection. In In Proceedings of the 33rd ACM International. 





Moussavou Boussougou, M. K. & Park, D. J. (2023). Attention-based 1d cnn-bilstm hybrid model enhanced with fasttext word embedding for korean voice phishing detection. Mathematics, 11(14), 3217. 





Shen, Z., Yan, S., Zhang, Y., Luo, X., Ngai, G., & Fu, E. Y. (2025). "it warned me just at the right moment": Exploring llm-based real-time detection of phone scams. arXiv preprint arXiv:2502.03964. 





Tan, Z. H. & et al. (2025). A review of deep learning based multimodal forgery detection for video and audio. Discover Applied Sciences, 7, 987. 





Wang, X., Yamagishi, J., Todisco, M., Delgado, H., Nautsch, A., Evans, N., ..., & Kinnunen, T. (2020). Asvspoof 2019: A large-scale public database of synthesized, converted, and replayed speech. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 2259–2275. 





Weller, O., Sperber, M., Gollan, C., & Kluivers, J. (2021). Streaming models for joint speech recognition and translation. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 1–11. 





Wickramasinghe, Y., Rathnayake, P., Madumal, O., Ramzan, R., & Galappaththi, K. (2022). A review on telecommunication frauds and fraud detection techniques. In Proceedings of the 1st Research Development and Innovation Conference. 





Zhang, B., Cui, H., Nguyen, V., & Whitty, M. (2025). Audio deepfake detection: What has been achieved and what lies ahead. Sensors, 25(7), 1989. 

