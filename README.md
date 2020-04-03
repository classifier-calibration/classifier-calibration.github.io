# Classifier Calibration:
## How to assess and improve classifier confidence and uncertainty

### A tutorial at ECML-PKDD 2020 

Peter Flach,
University of Bristol, UK,
[Peter.Flach@bristol.ac.uk ](mailto:Peter.Flach@bristol.ac.uk),
[www.cs.bris.ac.uk/~flach/ ](www.cs.bris.ac.uk/~flach/)

Miquel Perello-Nieto,
University of Bristol, UK,
[miquel.perellonieto@bristol.ac.uk](mailto:miquel.perellonieto@bristol.ac.uk),
[https://www.perellonieto.com/](https://www.perellonieto.com/)

Hao Song,
University of Bristol, UK,
[hao.song@bristol.ac.uk](mailto:hao.song@bristol.ac.uk)

Meelis Kull,
University of Tartu, Estonia,
[meelis.kull@ut.ee](mailto:meelis.kull@ut.ee)

Telmo Silva Filho,
Federal University of Paraiba, Brazil,
[telmo@de.ufpb.br](mailto:telmo@de.ufpb.br)


# Abstract 

This tutorial introduces fundamental concepts in classifier calibration and gives an overview of recent progress in the enhancement and evaluation of calibration methods. Participants will learn why some training algorithms produce calibrated probability estimates and others don't, and how to apply post-hoc calibration techniques in order to improve the probability estimates in theory and in practice, the latter in a Section dedicated to Hands-On explanations. Participants will furthermore learn how to test if a classifier’s outputs are calibrated and how to assess and evaluate probabilistic classifiers using a range of evaluation metrics and exploratory graphical tools. Additionally, participants will obtain a basic appreciation of the more abstract perspective provided by proper scoring rules, and learn about related topics and some open problems in the field.

# Description

This tutorial aims at providing guidance on how to evaluate models from a calibration perspective and how to correct some distortions found in a classifier’s output probabilities/scores. We will cover calibrated estimates of the posterior distribution, post-hoc calibration techniques, calibration evaluation and some related advanced topics. Among the main intended learning outcomes are the following. Participants will: 

*   understand the main advantages of calibrated classifiers, particularly in relation to changing misclassification costs and changing class priors;
*   learn the major definitions of calibrated outputs in the field, as well as their relative relationship;
*   understand why some training algorithms produce calibrated probability estimates and others don't, and be able to apply calibration techniques in post-processing; 
*   have a grasp of basic methods to evaluate probabilistic classifiers and be able to use graphical tools such as reliability diagrams and cost curves to analyse their performance in more detail;  
*   be introduced to a range of established and recently developed techniques to quickly obtain better calibrated results from trained models;
*   learn how to use available calibration tools and the steps needed to train and evaluate a calibrated model in a Hands-On approach;
*   learn about a few advanced and related topics and open problems, such as alternative views of calibration and other forms of uncertainty.

The tutorial will include practical demonstrations of some of the material by means of Jupyter Notebooks which will be made available online to participants in advance.

This tutorial will benefit machine learning researchers of different abilities and experience. PhD students and machine learning novices will profit from a gentle introduction to classifier calibration and achieve a better understanding of why good classifier scores matter. Only basic machine learning knowledge is expected (at the level of Mitchell or Witten & Frank or Peter Flach’s book, among others). More experienced machine learning researchers, who may already be familiar with the more basic material on calibration, will benefit from the comprehensive perspective that the tutorial provides, and perhaps be encouraged to tackle some of the open problems in their own research. 

This tutorial is relevant to the ECML-PKDD community, with previous work related to calibration having been published and presented in past conferences [16, 18], including a best paper award in the ECML-PKDD 2014 conference for the paper on reliability maps by Kull and Flach [16]. Calibration and uncertainty quantification are also receiving growing attention among other major ML / AI conferences, such as ICML, NeurIPS (see figure below) and AISTATS, demonstrating that there is growing interest on the interpretability of classification model outputs in order to make better informed decisions.

# Outline

This is a three and a half hour tutorial divided into five sections, with the final Section devoted to a recap and discussion of open problems. We first give the planned schedule with main contents in the following table. Then we briefly describe each of the five Sections below in separate paragraphs.


<div>
<table>
  <tr>
   <td><strong>Duration </strong>
   </td>
   <td><strong>Section (presenter)</strong>
   </td>
   <td><strong>Topics covered</strong>
   </td>
  </tr>
  <tr>
   <td>45min 
   </td>
   <td>1)
<p>
Intro:
<p>
The concept of calibration
<p>
(Peter Flach)
   </td>
   <td>
<ul>

<li>Motivation with different types of classifier outputs
</li>
<li>Optimal decision making
</li>
<li>Sources of miscalibration
</li>
<li>Visualisation of calibration and miscalibration
</li>
<li>Simple methods to obtain calibrated probabilities (binning methods)
</li>
<li>Notions of multi-class calibration: from weakest to strongest
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>45min 
   </td>
   <td>2)
<p>
Evaluation metrics and
<p>
proper scoring rules
<p>
(Telmo Silva Filho)
   </td>
   <td>
<ul>

<li>Classification calibrated loss, Ranking calibrated loss and proper losses
</li>
<li>Decompositions: calibration/refinement and others
</li>
<li>Calibration measures: conf-ECE, cw-ECE, total variation distance
</li>
<li>Hypothesis tests for calibration
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>30min
   </td>
   <td>BREAK
   </td>
   <td>
<ul>

<li>Hands-On material available online to download
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>60min 
   </td>
   <td>3)
<p>
Calibrators
<p>
(Hao Song)
   </td>
   <td>
<ul>

<li>Non-parametric approaches 
</li>
<li>Parametric approaches
</li>
<li>General practice
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>30min
   </td>
   <td>4)
<p>
Hands-On 
<p>
(Miquel Perello-Nieto)
   </td>
   <td>
<ul>

<li>Available packages for calibration
</li>
<li>Non-neural and neural demonstrations
</li>
<li>The pipeline on how to train and evaluate classifiers and calibrators
</li>
<li>Visualisation tools
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>30min
   </td>
   <td>5)
<p>
Advanced topics and conclusion
<p>
(Peter Flach)
   </td>
   <td>
<ul>

<li>Limitations and open problems
</li>
<li>The cost-sensitive perspective as an alternative view of calibration
</li>
<li>Regression / Distribution calibration
</li>
<li>Related uncertainty concepts
</li>
</ul>
   </td>
  </tr>
</table>
</div>



The **five sections** are described in more detail in the following paragraphs. 

**1) The concept of calibration:** We start by introducing the concept of calibration. A predictive model is well-calibrated if its predictions correspond to observed distributions in the data. In particular, a probabilistic classifier can be said to be well-calibrated if, among the instances receiving a predicted probability vector _p_, the class distribution is approximately given by _p_. This Section will cover different notions of calibration and how it can help with **optimal decision making**; exemplify what are some possible **sources of miscalibration** by means of simple examples; define the binary and multi-class scenarios together with corresponding **visualisations**; demonstrate how to obtain calibrated probabilities with **simple techniques**, such as binning methods; and propose different **notions of multi-class calibration** from the weakest to the strongest (confidence-calibrated, classwise-calibrated and multiclass-calibrated).

**2) Evaluation metrics and proper scoring rules:** Here, participants will learn how to evaluate the **quality of classifier outputs** from the calibration perspective. We start by introducing **different losses** starting from classification losses (eg. accuracy) and ending with proper losses (eg. Brier score). We will show that proper losses can be **decomposed** into calibration, refinement and other losses. We then explain the different versions of the **expected calibration error** (ECE), showing how they correlate with various levels of calibration and how they are related to some of the visualisation tools introduced in Section 1. We end this Section with **hypothesis tests** for calibration, with the null hypothesis being that the scores given by a model are already calibrated.

**3) Calibrators:** This section introduces both well-known and recently developed state-of-the-art techniques to improve the level of calibration, as well as practical details when applying them. The techniques are organised into two categories: (1) **non-parametric** approaches that can particularly benefit from large training sets; and (2) **parametric** approaches that are relatively fast to learn and apply, and show good performance. Established calibration methods include logistic calibration and the ROC convex hull method (also known as pair-adjacent-violators or isotonic regression), while recently introduced calibration methods include beta calibration, which is designed for probabilistic binary classifiers; Dirichlet calibration, the natural extension of beta calibration to the multi-class scenario; and temperature scaling, vector scaling and matrix scaling, which were particularly designed for deep neural networks. We conclude this section by giving **general advice** about the application of different calibration methods, including regularisation techniques.

**4) Hands-On course:** This section consists of a Hands-On Course in which we cover existing **Python packages** and implementations of calibration techniques, while providing a series of Jupyter Notebooks available to be followed or run by the participants. The material will be made available beforehand and announced during the break for download or to be run online with Google Colab. The content will focus on a **full pipeline** on how to train and evaluate classifiers and calibrators for **neural and non-neural** models, the process to produce **statistical comparison** of calibration methods on several datasets, and also covers **visualisation tools** which will provide better insights into the strengths and weaknesses of uncalibrated classifiers and their calibrated counterparts (eg. reliability diagrams in a multi-class scenario).

**5) Advanced topics:** To conclude the tutorial, we will discuss **open problems** on calibration, and recent methods that may lead to innovative solutions. This will include the **cost-sensitive perspective** as an alternative view of calibration, with different scoring rules giving rise to different cost-based assumptions. We will also briefly discuss **calibration for regression** tasks and other related tasks in **uncertainty quantification**, such as out of distribution (OOD) samples and error decomposition into epistemic and aleatoric losses.


# Presenters 

While the presenters are based at three different institutions in as many countries, they have a well-established and ongoing track record of working together. They also all have good to very close familiarity with the ECML-PKDD conference series. 

**Peter Flach** ([Peter.Flach@bristol.ac.uk](mailto:Peter.Flach@bristol.ac.uk)) presents Sections 1 (Introduction) and 5 (Conclusion). He is Professor of Artificial Intelligence at the University of Bristol and has over 25 years experience in machine learning and data mining, with particular expertise in mining highly structured data and the evaluation and improvement of machine learning models using ROC analysis and associated tools. He was PC co-chair of KDD'09 and ECML-PKDD'12 and authored (_Machine Learning: the Art and Science of Algorithms that Make Sense of Data_, Cambridge University Press, 2012, [mlbook.cs.bris.ac.uk](mlbook.cs.bris.ac.uk)) which has to date sold about 15,000 copies and has been translated into Russian, Mandarin and Japanese. Since 2010 he has been Editor-in-Chief of _Machine Learning_. He is a Fellow of the Alan Turing Institute and President of the European Association for Data Science. He has taught tutorials on inductive logic programming, ROC analysis and machine learning at ACML, ECAI, ECML-PKDD, ICML, UAI, and various summer schools. His current Google Scholar profile ([https://scholar.google.com/citations?user=o9ggd4sAAAAJ](https://scholar.google.com/citations?user=o9ggd4sAAAAJ)) lists over 250 publications with over 11,000 citations and a Hirsch-index of 51. 

**Telmo Silva Filho** (telmo@de.ufpb.br) presents Section 2 (Evaluation metrics). He is an adjunct professor at the Department of Statistics of the Federal University of Paraiba (Brazil) and has over 10 years of experience in machine learning and data science, particularly complex data representations, optimisation, model evaluation and classifier calibration.

**Hao Song** ([hao.song@bristol.ac.uk](mailto:hao.song@bristol.ac.uk)) presents Section 3 (Calibration methods). He is currently a postdoctoral researcher at the University of Bristol. His research interests are mainly on quantifying different types of uncertainties within the machine learning pipeline, particularly for different kinds of probabilistic outputs and corresponding evaluation metrics. 

**Miquel Perello-Nieto** ([miquel.perellonieto@bristol.ac.uk](mailto:miquel.perellonieto@bristol.ac.uk)) presents Section 4 (Hands-On). He is a Research Associate at the University of Bristol and has over 8 years experience in machine learning, artificial intelligence and data mining. He has held Research positions for the last 5 years while pursuing a PhD in Computer Science, and he has started and organised the [PyData Bristol Meetup](https://www.meetup.com/PyData-Bristol/) for the last 2 years which currently has ~900 members, and he leads monthly talks and workshops with an attendance of [~100 people per event](https://www.meetup.com/PyData-Bristol/events/past/). His research interests are on uncertainty evaluation of probabilistic classifiers, and its applications to semi-supervised learning and learning in the presence of weak labels.

**Meelis Kull** ([meelis.kull@ut.ee](mailto:meelis.kull@ut.ee)) is not currently planning to attend the conference due to possible calendar conflicts. He will however take active part in the preparation and organisation of the material. He is an associate professor at the University of Tartu, Estonia. His research interests cover topics in machine learning and artificial intelligence. He has recently been working on evaluation, uncertainty quantification and calibration of machine learning models, and on machine learning methods that tolerate varying deployment context.


# Previous tutorials 

Presenter Peter Flach has given many tutorials, courses and lectures on machine learning, including the following on evaluation methods, ROC analysis, probability estimation and context-aware knowledge discovery (presented with Meelis Kull and others) which are related (but not identical) to this proposal, which has not been presented in this form before.


- ICML'04 tutorial: _The Many Faces of ROC Analysis in Machine Learning_: [http://www.cs.bris.ac.uk/~flach/ICML04tutorial/](http://www.cs.bris.ac.uk/~flach/ICML04tutorial/) (69 Google Scholar citations) 
- UAI’07 tutorial: _ROC Analysis for Ranking and Probability Estimation_: [http://www.auai.org/uai2007/tutorials.html#roc](http://www.auai.org/uai2007/tutorials.html#roc) 
- ECAI'12 tutorial: _Unity in Diversity: The Breadth and Depth of Machine Learning Explained for AI Researchers_: [http://www.lirmm.fr/ecai2012/index.php?option=com_content&view=article&id=96 &Itemid=104](http://www.lirmm.fr/ecai2012/index.php?option=com_content&view=article&id=96) 
- INIT/AERFAI Summer School on Machine Learning 2013 and 2017: _ROC Analysis and Performance Evaluation Metrics_: [http://www.init.uji.es/school2013/lecturers.html](http://www.init.uji.es/school2013/lecturers.html)
- ECML-PKDD'16 tutorial: _Context-Aware Knowledge Discovery: Opportunities, Techniques and Applications_: [https://docs.google.com/presentation/d/1Q1_Wh8dcMDCH5DGuSxs_bieyIl8oDubmiYuZf8l9qu4/pub?slide=id.p](https://docs.google.com/presentation/d/1Q1_Wh8dcMDCH5DGuSxs_bieyIl8oDubmiYuZf8l9qu4/pub?slide=id.p)

Audience sizes for these tutorials are not available, but taking into account the number of participants at ECML-PKDD 2019 (800 attendees) and the proportion of papers at similar conferences/proceedings mentioning calibration (around 5% at NeurIPS 2019), we estimate an audience size of 50-100 participants. 


# Required technical equipment 

Participants will be able to follow the full tutorial just by means of the presenter’s projector screen. However, most of the material will be available online and some parts (eg. the Hands-On course) will be in the form of Jupyter Notebooks in case some of the participants want to run the Notebooks by themselves, or run them online with Google Colab. For that reason, although the tutorial content can be delivered without an internet connection, it would be advisable to have sufficient bandwidth, in case participants want to follow some parts of the tutorial in a more proactive manner.

# References

An initial list in chronological order is given below. This includes work on forecasting and proper scoring rules [1,11]; foundational work on cost-sensitive learning and calibration [2-4,6-7]; ROC analysis and cost curves [5, 9-10]; empirical analysis [8, 12]; and recent advances [13-25]. 

1. Glenn Brier. Verification of forecasts expressed in terms of probabilities. Monthly Weather Review, 78:1–3, 1950.
2. John Platt. Probabilities for SV machines. In A. Smola, P. Bartlett, B. Scho ̈lkopf, and D. Schuurmans, editors, Advances in Large Margin Classifiers, pages 61–74. MIT Press, 2000.
3. Charles Elkan. The foundations of cost-sensitive learning. In Proc. 17th Int. Joint Conf. on Artificial intelligence (IJCAI’01), pages 973–978. Morgan Kaufmann, 2001.
4. Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In Proc. 18th Int. Conf. on Machine Learning (ICML’01), pages 609–616, 2001.
5. Foster Provost and Tom Fawcett. Robust classification for imprecise environments. _Machine Learning_, 42(3):203–231, March 2001.
6. Barbara Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proc. 8th Int. Conf. on Knowledge Discovery and Data Mining (KDD’02), pages 694–699. ACM, 2002.
7. Foster Provost and Pedro Domingos. Tree induction for probability-based ranking. Machine Learning, 52(3):199–215, 2003.
8. Alexandru Niculescu-Mizil and Rich Caruana. Predicting good probabilities with supervised learning. In Proc. 22nd Int. Conf. on Machine Learning (ICML’05), pages 625–632, 2005.
9. Chris Drummond and Robert Holte. Cost curves: An improved method for visualizing classifier performance. Machine Learning, 65(1):95–130, 2006.
10. Tom Fawcett and Alexandru Niculescu-Mizil. PAV and the ROC convex hull. Machine Learning, 68(1):97–106, 2007.
11. Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477):359–378, 2007.
12. Chris Bourke, Kun Deng, Stephen Scott, Robert Schapire, and N.V. Vinodchandran. On reoptimizing multi-class classifiers. Machine Learning, 71(2-3):219–242, 2008.
13. José Hernández-Orallo, Peter Flach, and Cesar Ferri. Brier curves: A new cost-based visualisation of classifier performance. In Proc. 28th Int. Conf. on Machine Learning (ICML’11), pages 585–592, 2011. 
14. José Hernández-Orallo, Peter Flach, and Cesar Ferri. A unified view of performance metrics: translating threshold choice into expected classification loss. Journal of Machine Learning Research, 13(1):2813–2869, 2012. 
15. Ming-Jie Zhao, Narayanan Edakunni, Adam Pocock, and Gavin Brown. Beyond Fano’s inequality: bounds on the optimal F-score, BER, and cost-sensitive risk and their implications. Journal of Machine Learning Research, 14(1):1033–1090, 2013. 
16. Meelis Kull and Peter Flach. Reliability Maps: A Tool to Enhance Probability Estimates and Improve Classification Accuracy. In: Calders T., Esposito F., Hüllermeier E., Meo R. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2014. Lecture Notes in Computer Science, vol 8725. Springer, Berlin, Heidelberg, 2014
17. Oluwasanmi O Koyejo, Nagarajan Natarajan, Pradeep K Ravikumar, and Inderjit S Dhillon. Consistent binary classification with generalized performance metrics. In Advances in Neural Information Processing Systems (NIPS’14), pages 2744–2752, 2014. 
18. Meelis Kull and Peter Flach. Novel decompositions of proper scoring rules for classification: Score adjustment as precursor to calibration. In Machine Learning and Knowledge Discovery in Databases (ECML-PKDD’15), pages 68–85. Springer, 2015. 
19. Peter Flach and Meelis Kull. Precision-recall-gain curves: PR analysis done right. In Advances in Neural Information Processing Systems (NIPS’15), pages 838–846, 2015. 
20. Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In AAAI Conference on Artificial Intelligence, 2015.
21. Mahdi Pakdaman Naeini and Gregory Cooper. Binary classifier calibration using an ensemble of near isotonic regression models. In 2016 IEEE 16th International Conference on Data Mining (ICDM), pages 360–369. IEEE, 2016.
22. Meelis Kull, Telmo M. Silva Filho, and Peter Flach.  Beyond sigmoids:  How to obtain well-calibrated probabilities from binary classifiers with beta calibration.Electron. J. Statist., 11(2):5052–5080,2017.
23. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On Calibration of Modern Neural Networks.InThirty-fourth International Conference on Machine Learning, Sydney, Australia, jun 2017.
24. Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, and Thomas Schön.  Evaluating model calibration in classification. In K. Chaudhuri and M. Sugiyama, editors, Proceedings of Machine Learning Research, volume 89 of Proceedings of Machine Learning Research, pages 3459–3467. PMLR, 16–18 Apr 2019.
25. Kull, M., Perello Nieto, M., Kängsepp, M., Silva Filho, T., Song, H. & Flach, P. Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration, 3 Sep 2019, NeurIPS 2019.
