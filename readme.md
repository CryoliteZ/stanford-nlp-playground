# Stanford NLP Playground

## Stanford Parser: Statistical Parser

### About [[Software Page](https://nlp.stanford.edu/software/lex-parser.html)]
A natural language parser is a program that works out the grammatical structure of sentences, for instance, which groups of words go together (as "phrases") and which words are the subject or object of a verb. Probabilistic parsers use knowledge of language gained from hand-parsed sentences to try to produce the most likely analysis of new sentences. These statistical parsers still make some mistakes, but commonly work rather well. Their development was one of the biggest breakthroughs in natural language processing in the 1990s.

### Pacakge Summary [[link](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/parser/lexparser/package-summary.html#package.description)]

#### Pacakge Name [edu.stanford.nlp.parser.lexparser.LexicalizedParser](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/parser/lexparser/LexicalizedParser.html)
This package contains implementations of 3 probabilistic parsers
* accurate unlexicalized probabilistic context-free grammar (PCFG) parser
* probabilistic lexical dependency parser
* factored, lexicalized probabilistic context free grammar parser, which does joint inference over the product of the first two parsers

### Dependency Parsing
#### Two main approaches:
* Grammar-based parsing
    * **Context-free dependency grammar ✔️**
    * **Lexicalized context-free grammars ✔️**
    * Constraint dependency grammar
* Data-driven parsing
    * Graph-based models
    * Transition-based models
    * Easy-first parsing
    * Hybrids: grammar+data-driven, ensembles, etc


### Papers
The factored parser and the unlexicalized PCFG parser are described in:

* Dan Klein and Christopher D. Manning. 2002. Fast Exact Inference with a Factored Model for Natural Language Parsing. Advances in Neural Information Processing Systems 15 (NIPS 2002). [[pdf](https://nlp.stanford.edu/~manning/papers/lex-parser.pdf)]

* Dan Klein and Christopher D. Manning. 2003. Accurate Unlexicalized Parsing. Proceedings of the Association for Computational Linguistics, 2003. [[pdf](https://nlp.stanford.edu/~manning/papers/unlexicalized-parsing.pdf)]

Chinese parser

* Roger Levy and Christopher D. Manning. 2003. Is it harder to parse Chinese, or the Chinese Treebank? ACL 2003 (Chinese parser) [[pdf](https://nlp.stanford.edu/pubs/acl2003-chinese.pdf)]

Grammatical relations output of the parser
* Marie-Catherine de Marneffe, Bill MacCartney and Christopher D. Manning. 2006. Generating Typed Dependency Parses from Phrase Structure Parses. LREC 2006. [[pdf](https://nlp.stanford.edu/pubs/LREC06_dependencies.pdf)]


### Usage
#### CLI
```
java -mx4g edu.stanford.nlp.parser.lexparser.LexicalizedParser \
-encoding utf-8 \
-outputFormat "wordsAndTags,penn,dependencies,typedDependencies" \
edu/stanford/nlp/models/lexparser/chineseFactored.ser.gz tw.txt > tw.out.txt
```
#### Code

```java
LexicalizedParser lp = LexicalizedParser.loadModel("edu/stanford/nlp/models/lexparser/chineseFactored.ser.gz");
List<CoreLabel> rawWords = SentenceUtils.toCoreLabelList(sentence);
Tree parse = lp.apply(rawWords);
TreebankLanguagePack tlp = lp.treebankLanguagePack();
GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
for(TypedDependency t:tdl){
    System.out.println(t.toString());
}
```

#### Models
![](https://i.imgur.com/zj6CA2E.png)
* edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz
* edu/stanford/nlp/models/lexparser/chineseFactored.ser.gz
* edu/stanford/nlp/models/lexparser/xinhuaFactoredSegmenting.ser.gz
* edu/stanford/nlp/models/lexparser/xinhuaFactored.ser.gz
* edu/stanford/nlp/models/lexparser/xinhuaPCFG.ser.gz 

The PCFG parsers are smaller and faster. But the **Factored parser is significantly better for Chinese**, and we would generally recommend its use.

```java
// old version 
old_model = "edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz"

// new version?
new_model = "edu/stanford/nlp/models/lexparser/chineseFactored.ser.gz"

```
#### Parser Explained
##### PCFG parsers
* Context: unlexicalized PCFG grammar + 1 parser. （ACL 2003 Accurate Unlexicalized Parsing paper）

* Process: 基於CFG樹的pattern matching > Typed Dependencies為 postprocessing（Marie-Catherine de Marneffe paper）

* Parsers產生Dependencies Relations是基於Phrase Structure (CFG) Parse，而非Dependency Parse。對於一個PCFG Parser ，始用 `getBestDependencyParse()`將得到一個`NULL`，因為沒有進行Dependency分析，但仍可以使用單獨的`GrammaticalStructure`從PCFG結取得Typed Dependencies

##### Factored parsers
* Context: 2 Grammars + 3 parsers

* Process: Simple PCFG parser -> Untyped Dependency parser -> 3rd parser 綜合考慮前兩個parser的performacne，獲取一個最佳分析結果。（NIPS Fast Exact Inference paper）

* Dependencies Relations通過由此分析產生的Phrase Structure Parse得到.對於一個Factored Parser接口,調用getBestDependencyParse()將得到最佳的Untyped Dependency Parse結果. 


> 1、PCFG具有最快的分析速度，Factored對中文分析性能有較大的提高（推薦使用）
2、Xinhua Grammars主要使用大陆简体新闻语料， Mixed包含了台灣、香港等繁體語料。
3、前4個分析器（Factored 2個+ PCFG 2個）均需要已分詞，最後的一個不需要
4、絕大多數的Dependency Relations編碼在ChineseGrammaticalRelations 類中（可進一步參考）

Reference: [Stanford FAQ](https://nlp.stanford.edu/software/parser-faq.shtml#y), [Blog CSDN](https://blog.csdn.net/allenshi_szl/article/details/6093582)

#### Character encoding
Easy to change with `-encoding` flag

#### Word segmentation
For best results, Stanford recommend that we first segment input text with a high quality word segmentation system ([Stanford Segmenter](https://nlp.stanford.edu/software/segmenter.html))

### Appendix
#### Stanford tags
Hackmd page [here](https://hackmd.io/UgRL3nj0SdaGUzLQPKCmzQ?view)

#### About CFG(context-free grammar) trees and parsing
https://www.nltk.org/book/ch08.html

#### Parsing Chinese text with Stanford NLP
http://michelleful.github.io/code-blog/2015/09/10/parsing-chinese-with-stanford/

http://fancyerii.github.io/books/stanfordnlp/




## Neural Network Dependency Parser
### About [[Software Page](https://nlp.stanford.edu/software/nndep.html)]
A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between "head" words and words which modify those heads. The figure below shows a dependency parse of a short sentence. The arrow from the word moving to the word faster indicates that faster modifies moving, and the label advmod assigned to the arrow describes the exact nature of the dependency.

![](https://i.imgur.com/0HaXm10.png)


We have built a **super-fast transition-based parser** which produces typed dependency parses of natural language sentences. The parser is powered by a neural network which **accepts word embedding inputs**, as described in the paper:

* Danqi Chen and Christopher Manning. 2014. A Fast and Accurate Dependency Parser Using Neural Networks. In Proceedings of EMNLP 2014. [[pdf](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf)]

#### Two Main Approaches
* Grammar-based parsing
    * Context-free dependency grammar
    * Lexicalized context-free grammars
    * Constraint dependency grammar
* Data-driven parsing
    * Graph-based models
    * **Transition-based models ✔️**
    * Easy-first parsing
    * Hybrids: grammar+data-driven, ensembles, etc

### Pretrained Models
Trained models for use with this parser are included in either of the packages. The list of models currently distributed is:
```
.../parser/nndep/english_UD.gz (default, English, Universal Dependencies)
.../parser/nndep/english_SD.gz (English, Stanford Dependencies)
.../parser/nndep/PTB_CoNLL_params.txt.gz (English, CoNLL Dependencies) 
.../parser/nndep/CTB_CoNLL_params.txt.gz (Chinese, **CoNLL Dependencies**)
```

The Chinese model uses **CoNLL Dependencies**.

### *A Fast and Accurate Dependency Parser Using Neural Networks* 
#### Intro
Old dependency parsers may suffer from the use of millions of mainly poorly estimated feature weights
* Sparsity
> The features, especially lexicalized features are highly sparse, and this is a common problem in many NLP tasks. The situation is severe in dependency parsing, because it depends critically on word-to-word interactions
> 
* Expensive feature computation
> Concatenate some words, POS tags, or arc labels for generating feature strings, and look them up in a huge table containing several millions of features. Computation is extremely slow

改使用dense features，並訓練一個用Nerual Network做的transition-based dependency parser來解決以上的問題
> The neural network learns compact dense vector representations of words, part-of-speech (POS) tags, and dependency labels

#### Transition-based Dependency Parsing
Transition-based dependency parsing 會根據當前的configuraion預測下一個transition，直到終止configuration.

做決策時採用Greedy，選當前每一步最佳的transition就行，這樣只損失了一點準確率，換來了速度的大幅度提升

##### Configuration
```
c = (s, b, A)
s = stack
b = buffer
A = a set of dependency Arcs
```
初始configuration
```
s = [root]
b = [w1..wn]
A = []
```

終止條件
```
S = [root]
b = []
```

##### Transition
$s_i (i = 1,2..)$ i-th top element on the stack 
$b_i(i = 1,2 ..)$ i-th element in buffer

##### 3 Types of transition
1. **LEFT-ARC(l)**
Adds an arc ![](https://i.imgur.com/VIcXf6Z.png)  with label `l` and removes `s2 `from the stack
Precondition: |s| ≥ 2
2. **RIGHT-ARC(l)**
Adds an arc ![](https://i.imgur.com/QaXD6si.png) with label `l` and removes `s1`from the stack
Precondition: |s| ≥ 2.
3. **SHIFT**
Moves `b1` from the buffer to the stack
Precondition: |b| ≥ 1

$N_{l}$為不同arc labels的個數，一個configuration對應的transition有$2N_{l}+1$種，也就是說每一步決策都是一個$2N_{l}+1$分類問題

##### Step-by-Step
![](https://i.imgur.com/YX2y1CD.png)

#### NN Model
![](https://i.imgur.com/e6YCEH5.png)


##### Model Input and Training
* word vector: $e^w_i \in R^{d}$
* fulling embedding matrix $E^w \in R^{d \times N_w }$, ${N_w}$為dictionary size
* POS tag: $e^t_i \in R^{d}$, ith POS tags
* Arc Label: $e^l_j \in R^{d}$, jth arch label
* POS embedding:  $E^t \in R^{d \times N_t }$, ${N_t}$為POS tag種類個數
* Arc Label embedding:  $E^l \in R^{d \times N_l }$，$N_l$ 為Arch label種類個數


我們選取一組set of elements(word, POS, label)， based on stack / buffer position(對預測有幫助的），定義為 $S^w$ $S^t$ 和 $S^l$



以上圖為例，若 $S^t$ = ![](https://i.imgur.com/dpLlfBz.png)，我們得到的值為`PRP`,`VBZ`,`NULL`,`JJ`

* $x^w$ :[ $e^w_{w_1}$, $e^w_{w_2}$, $e^w_{w_3}$, ... $e^w_{w_n}$], where $S^w$ = ${\{w_1, ... w_{n_w} \}}$, $n_w$ number of choosen words
* $x^t$ :[ $e^t_{t_1}$, $e^t_{t_2}$, $e^t_{t_3}$, ... $e^t_{t_n}$], where $S^t$ = ${\{t_1, ... t_{n_t} \}}$, $n_t$ number of choosen tags
* $x^l$ :[ $e^l_{l_1}$, $e^l_{l_2}$, $e^l_{l_3}$, ... $e^l_{l_n}$], where $S^l$ = ${\{l_1, ... l_{n_l} \}}$, $n_l$ number of choosen labels

從configuration中提取的特徵，將單詞、詞性、已產生的arch label三種輸入的embedding cancat起來 

##### Pre-defined Set

$S^w$: Word Features (n=18)

![](https://i.imgur.com/zIWMSn1.png)

$S^t$: POS tag Features  (n=18)

![](https://i.imgur.com/kde3umu.png)

$S^l$: Arch label Features (n=12)

![](https://i.imgur.com/BeLNJB9.png)

$lc_1$代表left-most children，$rc_1$代表right-most children

##### Cube activation function

![](https://i.imgur.com/GQuhITF.png)

##### Objective function

<img src="https://i.imgur.com/WPe99BD.png" height="200px">

#### Tricks
1. 使用cube而不用傳統的relu或sigmoid，這樣的設計讓三種不同的embedding都能互相乘積
![](https://i.imgur.com/FT1s5mU.png)

2. POS and label embedding
Paper宣稱為第一個使用POS tag和label embedding來取代descrete representations。合理的推測為，某些tag feature如NN(singular noun)距離應該會離NNS(plural noun)近。用dense representation來capture這樣的semantic meanings

3. 提前計算好高頻率單詞的embedding和權重矩陣相乘，這樣後面需要計算的時候只需要去裡面找，減少重複計算，節省時間

#### 與傳統方法比較
上面可知神經網絡為48個feature，因為神經網絡是非線性，包括立方函數都能幫助提取各個特徵直接的隱含聯繫，比如是否共同出現這種。但是傳統的方法只能人工將這48個特徵各種組合，表示一起出現的情況，那樣就會出現成千上百種組合，即成千上百種特徵，但是每個句子可能只出現了一兩種，這樣特徵向量（0101 one-hot vector）就會非常稀疏。而且統計起來也很耗時，關鍵是人工設計的特徵容易漏掉一些資訊，換一種場景，特徵就得重新設計
> 引述 manmanxiaowugun@CSDN


#### Results
* Effeciency & Accuracy beats traditional methods
![](https://i.imgur.com/iSs2qwq.png)

![](https://i.imgur.com/awl1tde.png)

* t-SNE of POS and label embeddings
![](https://i.imgur.com/pIJFzWk.png)

Found some semantic information in these embeddings

* What do $W^w_1$, $W^t_1$, $W^l_1$ capture?
將隱層的每一個神經元作為一個特徵（feature），分別看這個神經元與輸入（word，pos，label）的權重矩陣大於0.2的部分

看圖中多數分佈在POS，說明POS特徵很重要。這個方法可以幫我們看出什麼特徵更重要，是勝過傳統方法好的地方

![](https://i.imgur.com/9oqgQmQ.png)

### Usage

#### CLI
```java
java edu.stanford.nlp.parser.nndep.DependencyParser -model modelOutputFile.txt.gz -textFile rawTextToParse -outFile dependenciesOutputFile.txt
```
#### Code
```java
MaxentTagger tagger = new MaxentTagger(taggerPath);
DependencyParser parser = DependencyParser.loadFromModelFile(modelPath);
List<TaggedWord> tagged = tagger.tagSentence(sentence);
GrammaticalStructure gs = parser.predict(tagged);
List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
for(TypedDependency t:tdl){
  System.out.println(t.toString());
}
```

### Slides
1. [Introduction to Dependency Parsing](https://cl.lingfil.uu.se/~nivre/docs/eacl1.pdf)
2. [Graph Based Parsing](https://cl.lingfil.uu.se/~nivre/docs/eacl2.pdf)
3. [Transition Based Parsing](https://cl.lingfil.uu.se/~nivre/docs/eacl3.pdf)
4. [Dependency Parsing](https://cl.lingfil.uu.se/~nivre/docs/ACLslides.pdf)


#### References
[manmanxiaowugun@CSDN | 依存句法分析—A Fast and Accurate Dependency Parser using Neural Networks](https://blog.csdn.net/manmanxiaowugun/article/details/85563822)

## TODO
* Issues about latex syntax
* More concise categories