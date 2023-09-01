Certainly! Here's a translated version of the provided text in English:

**Title:** Intrusion Detection Using Machine Learning

**Introduction:**

These methods of analysis are effective for detection based on well-known parameters, but they have several limitations. Firstly, they become obsolete when dealing with new types of intrusions or attack patterns that have not been previously described. Additionally, these methods can only model simple recognition patterns, limiting their applicability to well-known and precisely modelable behaviors.

One solution to this limitation is the use of self-learning systems capable of identifying previously unknown threats. This is due to their ability to differentiate between "normal" and "abnormal" traffic by learning from supervised network data. Machine learning methods can learn complex behaviors of a system. By learning entire classes of normal traffic and attacks, trained classifiers have the potential to detect irregularities and previously unknown attacks. Moreover, machine learning methods promise to provide a real-time solution for detecting potential attacks so that countermeasures can be taken promptly.

Furthermore, these machine learning solutions are dynamic, unlike the static rule-based network behavior analysis commonly used.

For the study and implementation of a machine learning solution, we drew inspiration from Ralf C. Staudemeyer's article titled "Applying Long Short-Term Memory Recurrent Neural Networks to Intrusion Detection." This paper provided detailed procedures and comprehensive results.

**Study:**

**1. KDD Cup '99 Datasets**

Another advantage of R.C. Staudemeyer's article is that it examines the "KDD Cup '99" dataset, which is arguably the most studied dataset for network intrusion detection. It was constructed in 1998 by DARPA and MIT Lincoln Labs and consists of a variety of simulated normal traffic and recorded intrusions over a 9-week period in a military-type local network.

During this period, the environment gathered information on nearly five million connections in the form of raw TCP dump data. Each connection was then labeled as either normal or with the name of the attack. They are represented in about 100 bits, and Figure 2 and Figure 3 provide an overview.

Finally, the attacks are grouped into four main types:
1. DoS (Denial of Services): Denial of service
2. Probe: Unauthorized surveillance
3. U2R (User to Root): Unauthorized access to privileges on the machine
4. R2L (Remote to Local): Unauthorized access from a local machine from a remote one

The entire dataset was divided into multiple datasets to have a training set and a validation set. This distinction will be detailed in the following section explaining the functioning of machine learning more precisely.

The dataset was indeed used for the Knowledge Discovery and Data Mining 1999 competition. The objective was to train a machine learning system to correctly predict the nature of connections provided with an initial dataset. Performance, meaning the percentage of correct predictions, was then evaluated on a second set of data.

One of the primary goals of the internship was to achieve a similar level of accuracy as the competition winners and R.C. Staudemeyer, who achieved even better results.

**2. Machine Learning:**

The fundamental difference between classic programming and machine learning is that in the former, the designer provides the machine with a set of rules (its program) that, when applied to provided data, generates a result. In the latter, the developer presents data to the machine along with the expected results on that data to generate rules. These rules can then be reused on new data to find results that match.

**Supervised Learning:**

The challenge of learning is to find the best possible rules with the aim that the system achieves a very high level of prediction. In other words, it must, from the provided data, be able to predict the correct results as frequently as possible. In the case of network intrusions, this involves "teaching" the system how to recognize a certain attack so that it can detect it the next time it encounters it.

This is where datasets come into play. They aim to provide information about an entity, in this case, a connection, and characterize it, in our case, determining the type of connection. When the programmer specifies the expected information at the system's output, it is supervised learning. Of course, the more data that uniquely characterizes the entity, the easier it is to identify it among others and therefore predict the correct output.

**Data Weighting:**

However, not all features are equally important for each piece of data. Some features may be specific to a certain type of attack. In such cases, these features will be given more importance than others. Similarly, there may be common identical information across all connections, constants. These pieces of information cannot be used to differentiate one connection from the others. Hence, some data will have significant importance or a strong weight, while others of lesser relevance will have a weak or even zero weight for constants. This is data weighting, as shown in Figure 5.

**Optimal Learning:**

For effective learning, data weighting should not be extreme. If the learning rules are too specific to the dataset used for training, they may not be suitable for new data. This is called overfitting. For example, if there is a majority of denial of service attacks in the dataset, parameters with specific values for this type of attack will have significant weight. This is done to best identify behaviors specific to these intrusions and recognize them better. However, there may be datasets with a different distribution of data, a majority of probe attacks, for example. Parameters with a high weight for denial of service attacks may not be the most important for the new dataset. That's why you should try not to learn "too much" from a single data sample. On the other hand, you should avoid "under-learning" as well.

The objective is to find the optimal learning interval to have good predictions on the training dataset while performing well with new data. That's why we need to introduce a second dataset called validation. This second dataset, also provided by DARPA in our case, is simply another part of the collected data. However, the distribution of connection types is deliberately different so that the system can ensure it remains general and not too specific to a particular distribution.

Figure 6 illustrates the evolution of the error rate in the system's predictions on the two datasets. It can be observed that the error rate decreases on both datasets until a certain point where it starts to rise for the validation set. This is when the system is overfitting to these particular data. Therefore, it is important to continuously check throughout the learning process if the system remains general, and the error rate for the validation set does not increase. Learning should be stopped just before this point to achieve optimal results.

**3. Neural Networks:**

**Principle of Artificial Neurons:**

Machine learning algorithms are based on the principle of the biological neuron found in our brains. An artificial neuron can be seen as a system that takes input data and produces an output similar to a mathematical function. An aggregation of functions is contained in this system. This aggregation can be more or less complex depending on the algorithm that transforms the input data. An example of the transformation of a connection from the KDD Cup '99 dataset through a neuron is shown in Figure 7.

