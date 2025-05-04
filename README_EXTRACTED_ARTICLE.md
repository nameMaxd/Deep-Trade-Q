Deep Reinforcement Learning Approach for
Trading Automation in The Stock Market
Taylan Kabbani1, Ekrem Duman 2
Department of Industrial Engineering, Ozyegin University, Istanbul, Turkey
1taylan.kabbani1@ozu.edu.tr
2ekrem.duman@ozyegin.edu.tr
2
2
0
2
l
u
J
5
]
R
T
.
n
i
f
-
q
[
1
v
5
6
1
7
0
.
8
0
2
2
:
v
i
X
r
a
Abstract—Deep Reinforcement Learning (DRL) algorithms can
scale to previously intractable problems. The automation of proﬁt
generation in the stock market is possible using DRL, by combining
the ﬁnancial assets price ”prediction” step and the ”allocation” step
of the portfolio in one uniﬁed process to produce fully autonomous
systems capable of interacting with its environment to make optimal
decisions through trial and error. This work represents a DRL model
to generate proﬁtable trades in the stock market, effectively overcom-
ing the limitations of supervised learning approaches. We formulate
the trading problem as a Partially Observed Markov Decision Process
(POMDP) model, considering the constraints imposed by the stock
market, such as liquidity and transaction costs. We then solve the
formulated POMDP problem using the Twin Delayed Deep Deter-
ministic Policy Gradient (TD3) algorithm reporting a 2.68 Sharpe
Ratio on unseen data set (test data) . From the point of view of stock
market forecasting and the intelligent decision-making mechanism,
this paper demonstrates the superiority of DRL in ﬁnancial markets
over other types of machine learning and proves its credibility and
advantages of strategic decision-making.
Keywords—Autonomous agent, Deep reinforcement
learning,
MDP, Sentiment analysis, Stock market, Technical indicators, Twin
delayed deep deterministic policy gradient
I. INTRODUCTION
T HE prime objective of any investor when investing in
any ﬁnancial market is to minimize the risk involved
in the trading process and maximize the proﬁts generated.
Investors can meet this objective by successfully predicting the
prices or trends of the market assets and optimally allocating
the capital among the selected assets. This process is very
challenging for a human to consider all relevant factors in
a complex and dynamic environment; therefore, the design
of adaptive automated trading systems capable of meeting
the investor’s objective and bringing more stagnant wealth
into the global market has been an intensive research topic.
Many efforts have been made to design such trading systems
in the past decade. The majority of these efforts focused on
using Supervised learning (SL) techniques [1], [2], [3], [4],
[9], which in essence train a predictive model (e.g., Neural
Network, Random Forest,...) on historical data to forecast the
trend direction of the market. Regardless of their popularity,
these techniques suffered from various limitations, leading to
sub-optimal results [5]. Reinforcement Learning (RL) offers
to solve the drawbacks of Supervised Learning approaches
in trading ﬁnancial markets by combining the ﬁnancial assets
price ”prediction” step and the ”allocation” step of the port-
folio in one uniﬁed process to optimize the objective of the
investor, where the trading agent (the algorithm) interacts with
the environment (the model) to take the optimal decision [6].
In addition, ﬁnancial data is highly time-dependent (function
of time), making it a perfect ﬁt for Markov Decision Processes
(MDP) [7], which is the core process of solving RL problems.
MDP captures the entire past data and deﬁnes the whole
history of the problem in just the agent’s current state, and
that’s highly crucial when it comes to modeling ﬁnancial
market data [8].
Most works that studied the RL’s applications in ﬁnancial
markets and particularly in trading stocks considered discrete
action spaces [9], [10], [11], [12], i.e., buy, hold, and sell a
ﬁxed number of shares to trade a single asset. In this work, a
continuous action space approach is adopted to give the trading
agent the ability to gradually adjust the portfolio’s positions
with each time step (dynamically re-allocate investments),
resulting in better agent-environment interaction and faster
convergence of the learning process. In addition, the approach
supports the managing of a portfolio with several assets instead
of a single one. We ﬁrst propose a novel formulation of the
stock trading problem or what is referred to as the trading
Environment as a Partially Observed Markov Decision Process
(POMDP) model considering the constraints imposed by the
stock market, such as liquidity and transaction costs. More
speciﬁcally, we design an environment that simulates the real-
world trading process by augmenting the state (observation)
representation with ten different technical indicators and sen-
timent analysis scores of news releases along with other state
components. We then solve the formulated POMDP problem
using the Twin Delayed Deep Deterministic Policy Gradient
(TD3) algorithm, which can learn policies in high-dimensional
and continuous action spaces like those typically found in the
stock market environment. Finally, we evaluate our proposed
approach by performing back-testing, which is the process
used by traders and analysts to assert the viability of a trading
strategy by testing it on historical data.
II. BACKGROUND AND RELATED WORK
A. MDP in Reinforcement Learning
In essence, Markov Decision Processes [13] (MDP) is used
to model stochastic processes containing random variables,
transitioning from one state to another depending on certain
assumptions and deﬁnite probabilistic rules. MDPs are a
perfect mathematical framework to describe the reinforcement
 
 
 
 
 
 

learning problem. In this framework, researchers call
the
learner or decision maker the agent and the surrounding which
the agent interacts with (comprising everything outside the
agent) the environment. The learning process ensues from
the agent-environment interaction in MDP, at each time step
t ∈ {1, 2, 3, ..., T } the agent receives some representation
(information) of its current state from the environment st ∈ S,
and on that basis selects an action at ∈ A to perform. One
step later, due to its action, the agent ﬁnds itself in a new state,
and the environment returns a reward Rt+1 ∈ R to the agent
as a feedback of its action’s quality [14].
B. The Objective of Reinforcement Learning
The objective of any RL problem is to maximize the
cumulative reward Gt it receives in the long run instead of
the immediate reward Rt
$$
E[Gt] = E[Rt+1 + Rt+2 + Rt+3 + ... + RT ]
$$
(1)
$$
In the above reward equation (Eq. 1), the term RT denotes
the reward received at the terminal state T. meaning that the
aforementioned equation is only valid when the problem at
hand is an Episodic task, i.e., ends in a terminal state T. For
the Continuous tasks i.e., no terminal state, T = ∞, a discount
factor gamma is introduced to Eq. 1 (0 ≤ γ ≤ 1):
$$
$$
Gt = Rt+1 + γRt+2 + γ2Rt+3 + ... + γk−1Rt+k + ....
$$
$$
=
$$
∞
(cid:88)
0
γkRt+k+1
(2)
C. Bellman Equations
Value functions are being used by almost all RL methods
to estimate how good (in terms of expected return) it is for
the agent to be in a given state or to perform an action in
a given state. This evaluation is being made based on the
future expected sum of rewards. Accordingly, value functions
are determined with respect to the future actions the agent
will take. We call a particular way of acting a Policy (π) [14]
which is a function that maps from environment’s states to
probabilities of selecting each possible action.
Bellman equations [15] are the fundamental property of
value functions used in dynamic programming as well as in
reinforcement learning to solve MDPs, and they are essen-
tial to understand how many RL algorithms work. Bellman
the value function of state s (Vπ(s))
equation states that
can be calculated by ﬁnding the sum over all possibilities
of expected returns, weighting each by its probability of
occurring following a policy π:
Vπ(s)
(cid:88)
$$
.
=
$$
π(a|s)
(cid:88)
(cid:88)
P (s(cid:48), r|s, a)[r + γVπ(s(cid:48))], ∀s ∈ S
a
s(cid:48)
r
(3)
In a similar way we deﬁne the action-value (qπ(s, a)) function
as:
$$
qπ(s, a) =
$$
(cid:88)
(cid:88)
s(cid:48)
r
P (s(cid:48), r|s, a)[r + γ
(cid:88)
a(cid:48)
π(a(cid:48)|s(cid:48))qπ(s(cid:48), a(cid:48))]
(4)
From Bellman equations (Eq. 3 and Eq. 4) we can derive
what is called The Bellman Optimality Equations. Intuitively,
the Bellman optimality equation expresses the fact that the
value of a state under an optimal policy (π∗) must equal the
expected return for the best action from that state [14], and
the optimal state-value function (V∗) equals to :
(cid:88)
(cid:88)
P (s(cid:48), r|s, a)[r + γV∗(s(cid:48))]
(5)
$$
V∗(s) = max
$$
a
s(cid:48)
r
Similarly, we deﬁne optimal action-value (q∗) function as:
(cid:88)
(cid:88)
$$
q∗(s, a) = max
$$
$$
qπ(s, a) =
$$
P (s(cid:48), r|s, a)[r + γ max
q∗(s(cid:48), a(cid:48))]
π
s(cid:48)
r
a(cid:48)
(6)
D. Taxonomy of RL Algorithms
RL algorithms are classiﬁed based on how to represent and
train the agent into three main approaches:
1) Critic-Only Approach: This family of algorithms learns
to estimate the value function (State-value function or action-
value function) by using what is called Generalized Policy
Iteration (GPI), this concept refers to the interaction of two
steps. The ﬁrst step is the policy-evaluation, the main goal
of this step is to collect information (value functions) under
the given policy to determine how good it is. The second
step is the policy-improvement, it is responsible of improving
the policy by choosing greedy actions with respect to the
value functions computed from the policy-evaluation step. The
two steps alternate in a consecutive manner until the value
functions and policies stabilize which means that the process
has reached an optimal policy as illustrated in Fig. 1.
Fig. 1. Generalized Policy Iteration [14]
We distinguish between two different ways the agent learns
the value function of the system. The ﬁrst way is Tabular
Solution Method where the value functions are represented as
arrays or tables and updated with more accurate values after
each iteration as the agent collects more experience. This way
of learning often ﬁnds exact solutions. However, it does not
generalize well, and the state and action spaces must be small
enough to be stored in tables.
The second possible way in the critic-only approach is
called Approximate Solution Method, which tends to generalize
better than Tabular Method but has lower discrimination, and it
is capable of learning the value function of systems with enor-
mous state and action spaces. Approximate methods achieve

this generalization by combining RL with supervised learning
algorithms. Deep Reinforcement Learning is considered an
approximate method that combines Neural Networks with RL.
Mnih et al. (2013) [16] is considered the father of DRL, where
he trained an agent of Deep Q-network (DQN) to play Atari
games, where pixels of the game screen were the input data
(state), and the directions of the joystick were actions. He
proved that DRL had outperformed all existing algorithms in
2015 [17].
2) Actor-Only Approach: All methods under the Critic-
Only approach rely on the GPI framework to learn approxi-
mate action values to infer a good policy. Actor-Only methods
(also called Policy Gradient Methods) estimate the gradient of
the objective which is maximizing rewards with respect to the
policy parameters and adjust the policy parameters θ based on
the estimate (Eq. 7). The parameterized policy function takes
state and action as an input and returns the probability of
taking that action in that state instead of taking the state only
as an input and returning the value function as Critic-Only
methods do. Note that in the bellow equation Gt represents
the expected reward at time t.
$$
θt+1 = θt + α∇ ln π(at|st, θt)Gt
$$
(7)
3) Actor-Critic Approach: In the Actor-Critic approach, the
actor’s job is to select actions at each time step to form the
policy, where the critic’s role is to evaluate these actions taken
by the actor. So the approach is gradually adjusting the policy
parameters θ of the actor to take actions that maximize the total
reward predicted by the critic. The TD error (δ) calculated by
the critic to evaluate the action is as follows:
$$
δ = Rt+1 + γ ˆV (st+1, w) − ˆV (st, w)
$$
(8)
The value function estimation of the current state ˆV (st, w) is
added as a baseline to make the learning faster. The parameter
θ of the actor is being adjusted in the way of maximizing the
total future reward from Eq. 7 and Eq. 8 we conclude the
equation used by the Actor-Critic to update the gradient at
each time step t as the following:
$$
θt+1 = θt+α∇ ln π(at|st, θt)(Rt+1+γ ˆV (st+1, w)− ˆV (st, w))
$$
Many researchers worked on improving the DQN algorithm.
Van Hasselt et al. (2015) [18] proposed to use two networks
instead of one Q-network to choose the action and the other
to evaluate the action taken to solve the deviation problem in
DQN. They called it Double-DQN. Lillicrap et al. (2018) [19]
built on the top of Double-DQN, an algorithm based on the
deterministic policy gradient (DDPG) that can operate over
continuous action spaces. The Twin Delayed Deep Determin-
istic Policy Gradient (TD3) algorithm which will be discussed
in section IV-A, was proposed by Fujimoto et al. (2019) [20]
to tackle the problem of the approximation error in DDPG.
E. RL in Finance
Bertoluzzo and Corazza (2012) [21] investigated the per-
formance of different RL algorithms in day-trading for one
selected Italian stock. Speciﬁcally, they compared the perfor-
mance of Q-learning, and Kernal-based reinforcement learn-
ing, concluding that Q-learning performance outperformed
Kernal-based RL. In a subsequent study (2014) [10], they
explored the effect of different reward functions such as Sharpe
ratio, average log return, and OVER ratio on the performance
of Q-learning. By trading six selected Italian stocks, they
reported that lagged return reward function has the best perfor-
mance. Instead of approximating a value function (critic-only),
Deng et al. (2017) [12] made one of the ﬁrst attempts on com-
bining Deep Learning with Recurrent Reinforcement Learning
to directly approximate a policy function. This approach is
called “deep recurrent reinforcement learning” (DRRL). In
their proposed method, ﬁrst, the DL part extracts 45 useful
features from the market to be used as state representative
in the environment. Secondly, they use a Recurrent Neural
Network (RNN) as a trading agent to interact with the deep-
generated state features and make decisions. To investigate
the potential advantage of Actor-Critic methods in solving
the day trading problem, Conegundes and Pereira (2020) [22]
used Deep Deterministic Policy Gradient (DDPG) algorithm
to solve the asset allocation problem. Considering different
constraints such as liquidity, latency, slippage, and transaction
costs, they back-tested their approach on the Brazilian Stock
Exchange datasets. They showed that their approach success-
fully obtained 311% cumulative return in three years with an
annual average maximum drawdown around 19%.
III. PROBLEM DESCRIPTION
The stock trading problem is being modeled as Partially
Observed Markov Decision Process (POMDP), which can be
formulated by describing its State Space, Action Space, and
Reward Function. The POMDP model of the problem is called
the trading environment, and it’s built to carefully mimic the
real-world trading process.
A. State Space
The state-space in the proposed environment is designed
to support multiple and single stock trading by representing
the state as (1+ 13 x N )-dimensional vector where N is the
number of assets we consider to trade in the market. Hence
the state space increases linearly with the number of assets
available to be traded.
+
There are two main parts of the state presentation. The ﬁrst
part is the Position State ∈ R1+N
which holds the current cash
balance and shares owned of each asset in the portfolio, and
the second part of the state is the Market Signals ∈ R12×N ,
which holds the necessary market features for each asset as a
tuple, these features are the required information provided to
the agent to make predictions of the market movement. The
ﬁrst type of information is based on the hypothesis of technical
analysis [23], which states that the future behavior of ﬁnancial
markets is conditioned on its past; hence technical indicators
are being used in the state space to help the agent interpret
the market behavior. The second type of information is based
on fundamental analysis [24], which studies everything from
the overall economy and industry conditions to news releases.

Therefore a Natural Language Processing (NLP) approach
is used to measure the general sentiment from the news
releases and integrate it with the state representation. The state
(observation) vector at each time step is provided to the agent
as follows:
$$
St = [[bt, ht], [{(Ci
$$
t, SSi
t, Ti
t)|i ∈ N }]]
Each component of the state space is deﬁned as follows:
• N ∈ ZN
+ : Number of assets in the portfolio.
• bt ∈ R+: The available cash balance in the portfolio at
time step t.
$$
• ht = {hi
$$
t , h1
of shares owned for each asset i in N at time step t.
$$
t|i ∈ N } = {h0
$$
t , ..., hN
+ : The number
t } ∈ ZN
• Ci
• SSi
t ∈ RN
+ : The close price of asset i in N at time step t.
t ∈ (−1, 0, 1): An integer 1, 0 or -1 to indicate the
sentiment of the news related to stock i at time step t.
• Ti
t: The 10 different Technical Indicators vector for asset
i in the portfolio at time step t using the past prices of
the asset in a speciﬁed look-back window (most common
window is 14 or 9).
$$
To demonstrate the state space, let’s assume that we have 3
different assets (N = 3) in the trading environment and an
initial capital of 1000$ to be invested, the state vector would
be a 40-dimensional vector and the initial state(s0) given by
the environment would be:
$$
$$
s0 = [[1000, 0, 0, 0][(p1
$$
0, SS1
0 , T 1
0 ), (p2
0, SS2
0 , T 2
0 ), (p3
0, SS3
0 , T 3
0 )]]
B. Action Space
The designed agent
in this study receives the state st
at each time step t as input and sends back action in the
range between 1 and -1 inclusive, at ∈ [−1, 1], the action
then is re-scaled using a constrain Kmax, which represents
transforming at
the maximum allocation (buy/sell shares),
to an integer K ∈ [−Kmax, ...., −1, 0, 1, ...., Kmax], which
stands for the number of shares to be executed, resulting in
decreasing, increasing or holding of the current position of the
corresponding asset [25]. There are two important conditions
regarding the action execution in our approach:
• If the current capital (cash) in the portfolio is insufﬁcient
to execute the buy action, the action will be partially
executed with what the current capital can buy of the
requested stock.
• If the number of shares for a speciﬁc asset (hi
t) in the
portfolio is less than the number of shares to be sold
t ∈ Z−), the agent will sell all the remaining shares of
(ai
this asset in the portfolio.
We can mathematically express the action space as the follow-
ing:
$$
At = {ai
$$
$$
t|i ∈ N } = {a0
$$
t , a1
t , ..., aN
t }
(9)
S.t.
t ∈ ZN
ai
$$
−Kmax ≤ ai
t if |ai
t = hi
ai
$$
t ≤ Kmax, ∀i ∈ N
t , ∀at ∈ Z−
t| > hi
Where:
* N : assets in the portfolio.
* At: the action vector sent by the agent to the environment.
* ai
t: the action (number of shares) to buy/sell for asset i
at time step t.
* Kmax: the maximum number of shares the agent can re-
* hi
allocate of an individual asset at each time step t.
t: the portfolio position (number of shares) of asset i at
time step t.
The action space depends on the number of assets available
in the portfolio N and it’s given as (2 × Kmax + 1)N ; hence
the action space increases exponentially by increasing N .
C. Reward Function
The difference between the portfolio value Vt at the end
of period t and the value at the end of previous period t − 1
represents the immediate reward r(s, a, s(cid:48)) received by the
agent after each action, and we denote the ﬁnal investment
return at a target time Tf as G.
$$
r(s, a, s(cid:48)) = Vt − Vt−1
$$
(10)
Where the portfolio value V at each time step is calculated
as:
Where:
$$
Vt = bt + ht.Ct
$$
(11)
* bt: the available cash balance in the portfolio at time step
t.
$$
* ht = {hi
$$
t|i ∈ N }: the position vector (number of shares
of each asset) at time step step t.
$$
* Ct = {C i
$$
t |i ∈ N }: the closing price of each asset in the
portfolio at time step t.
The transition cost can be represented in many different
ways in real life, and it varies from one broker to another.
To better simulate the real-world trading process in the stock
market, transaction costs (i.e., commission fees) are incorpo-
rated into the immediate reward (r(s, a, s(cid:48))) calculation. In
this study, we set the commission as a ﬁxed percentage of
the total closed deal cash amount, where dbuy represents the
commission percentage when buying is performed, and dsell
is the commission percentage for selling:
t , d1
$$
dt = {di
$$
$$
t|i ∈ N } = [d0


$$
dbuy,
0,
dsell,

t , ..., dN
t ]
if ai
if ai
if ai
$$
t > 0
t = 0
t < 0
$$
where : di
$$
t =
$$
The commission vector dt is incorporated into the immediate
reward function by excluding the commission amount paid
from the portfolio value calculated in Eq. 11, so the agent
would avoid excessive trading that results in a high commis-
sion rate and therefore avoids a negative reward:
$$
Vt = bt + ht.Ct − ht.(Ct−1 ◦ dt)
$$
(12)
In the above equation, the amount paid for the commission is
calculated by taking the Hadamard product of the commission
vector dt and the closing price of the previous period Ct−1,

that’s because the action of buying/selling occurred on the
previous state and therefore commission should be calculated
using the closing prices on that state.
overestimation bias, thus reducing the accumulation of errors
in the learning process by introducing three main components
to DDPG:
D. Environment Constraints and Assumptions
We impose the following constraints and assumptions on
the MDP environment for two main reasons. First, to idealize
and simplify the complex ﬁnancial market systems (e.g., via
liquidity assumption) without losing the nature of the problem,
and the second reason is to make the model closer to a real-
world situation.
1) Non-Negative Balance Constraint: The cash balance in
any state is not allowed to be negative, bt > 0. Therefore, the
actions should not result in a negative cash balance, to achieve
that, the environment prioritizes the execution of sell actions
(at < 0) in the action vector At (Eq. 9) to guarantee cash
liquidity in the portfolio so buy actions (at > 0) would be
fulﬁlled afterward. If the buy action still results in a negative
balance (i.e., not enough cash to fulﬁll the action), it is fulﬁlled
partially with what remains in the portfolio’s cash balance.
2) Short-Selling Constraint: Short selling is prohibited in
the designed environment, all portfolio’s positions must be
strictly non-negative:
$$
ht = {hi
$$
$$
t|i ∈ N } = {h0
$$
t , h1
t , ..., hN
t } ∈ ZN
+
3) Zero Slippage Assumption: When the market volatility
is high; slippage occurs between the price at which the trade
was ordered and the price at which it’s completed [26]. In this
study, the market liquidity is assumed high enough to meet the
transaction at the same price when it was ordered [27]. This
assumption is mostly valid in a real-world trading environment
when trading in big stock markets.
4) Zero Market Impact:
In ﬁnancial markets, a market
participant impacts the market when it buys or sells an asset
which causes the price change. The impact provoked by the
agent in this study is assumed to have no effect on the market
when it performs its actions. This assumption is mostly true
even in real-life trading when the market volume is big enough
to make the individual investment insigniﬁcant [27].
IV. DETAILS OF IMPLEMENTATION
A. The Trading Agent
Actor-Critic-based algorithms successfully solved the con-
tinuous action space utilizing function approximation and pol-
icy gradient methods. One of the most famous actor-critic, off-
policy algorithms is the Deep Deterministic Policy Gradient
algorithm (DDPG) [28]. Still, despite the excellent perfor-
mance DDPG achieved in continuous control problems, it has
a signiﬁcant drawback similar to many RL algorithms, which
is the overestimation of action values (maxa Q(st+1, at+1)) as
a result of function approximation error. This overestimation
bias is unavoidable in RL as we use estimates instead of
ground truth in the learning process. In this study, as our prob-
lem has a continuous space of actions, we use Twin Delayed
Deep Deterministic Policy Gradient (TD3) [20] algorithm,
which is a direct successor of DDPG but with improvements
to tackle the overestimation problem. TD3 can reduce the
1) Clipped Double Critic Networks: The ﬁrst component
added is a novel clipped variant of Double Q-learning
[18] to replace the single critic. Using two different and
separate critic networks to make an independent estimate
of the value function can be used to make unbiased
estimates of the actions selected using the opposite value
estimate. TD3 uses a clipped double Q-learning instead
of the traditional one used in Double Q-learning where
it takes the smallest value of the two critic networks
estimates, that is, if we use the traditional Double Q-
learning in actor-critic methods, the policy and target
networks are updated so slowly that they make similar
estimates and offered slight improvement.
2) Delayed Updates: The second component is added to
solve the residual error accumulation formed due to
the learning process without a ﬁxed target (estimates
instead). In Critic-Actor methods, this accumulation of
errors is ampliﬁed due to the interaction between the
policy (actor) and value (critic) networks, where the
policy gradient is maximized over the value estimate.
Delaying the policy network update, i.e., updating it
less frequently than the value network, allows the value
network to stabilize before it can be used to update
the policy gradient. This results in a lower variance of
estimates and, therefore, better policy.
3) Target Policy Smoothing Regularization: The ﬁnal com-
ponent is applying a regularization strategy to the target
policy by adding a small random noise and averaging
over mini-batches. This is important to reduce the vari-
ance of the target values when updating the critic, which
causes by overﬁtting spikes in the value estimate.
The agent in this paper performs daily trading operations
and to aid the agent to understand its environment (the stock
market), we augmented the state representation of ten different
technical indicators and news sentiment scores.
B. Technical Indicator
We used the ten most famous indicators used by technical
traders when trading in the stock market [23], we describe
them brieﬂy as follows:
1) Relative Strength Index (RSI) ∈ RN
+ : A momentum
indicator to measure the magnitude of recent price
changes and identify overbought or oversold conditions
in the stock price.
2) Simple Moving Average (SMA) ∈ RN
+ : An important
indicator to identify current price trends and the potential
for a change in an established trend.
3) Exponential Moving Average (EMA) ∈ RN
+ : Like
SMA, EMA is a technical indicator used to spot cur-
rent trends over time. However, EMA is considered an
improved version of SMA by giving more weight to the
recent prices considering old price history less relevant;
therefore it responds more quickly to price changes than
SMA.

Algorithm: Twin Delayed Deep Deterministic Policy
Gradient (TD3) [20]
1. Initialization
Critic networks Q(s, a|w1), Q(s, a|w2) and actor
π(s|θ), randomly, with weights W1, W2 and θ.
Target networks Q(cid:48)
W (cid:48)
1 ←− W1, W (cid:48)
Replay buffer D
2 and π(cid:48) with weights
2 ←− W2, θ(cid:48) ←− θ
1, Q(cid:48)
$$
2. foreach t=1 to T do
$$
$$
Initialize a random process N for action
exploration
Select action with exploration noise
a ∼ π(s|θ) + (cid:15), (cid:15) ∼ N (0, σ)
Observe reward r and next state s(cid:48)
Store transition tuple (s, a, r, s(cid:48)) in D
Sample mini-batch of N transitions (s, a, r, s(cid:48))
from D
˜a ← π(s(cid:48)|θ) + (cid:15), (cid:15) ∼ clip(N (0, ˜σ), −c, c)
y ← r + γ mini=1,2 Q(s(cid:48), ˜a|wi)
Update critics
Wi ← arg minWi N −1 (cid:80)(y − QWi(s, a))2
if t mode d then
$$
$$
Update θ by the deterministic policy gradient:
∇θJ(θ) =
N −1 (cid:80) ∇aQW1(s, a)|a=πθ(s)∇θπθ(s)
Update target networks:
i ← τ Wi + (1 − τ )W (cid:48)
W (cid:48)
i
θ(cid:48) ← τ θ + (1 − τ )θ(cid:48)
$$
4) Stochastic Oscillator (%K) ∈ RN
+ : It’s a momentum
indicator comparing the closing price of the stock to a
range of its prices in a look-back window period W.
5) Moving Average Convergence/Divergence (MACD) ∈
RN : Is one of the most used momentum indicators to
identify the relationship between two moving averages
of the stock price and it helps the agent to understand
whether the bullish or bearish movement in the price is
strengthening or weakening [29].
6) Accumulation/Distribution Oscillator (A/D) ∈ RN :
A volume-based cumulative momentum indicator that
helps the agent to assess whether the stock is being
accumulated (bought) or distributed (sold) by measuring
the divergences between the volume ﬂow and the stock
price.
7) On-Balance Volume Indicator (OBV) ∈ RN : Another
volume-based momentum indicator that uses volume
ﬂow to predict the changes in stock price [30]:
8) Price Rate Of Change (ROC) ∈ RN : A momentum-
based indicator that measures the speed of stock price
changes over the look-back window W.
9) William’s %R ∈ RN
+ : Known also as Williams Percent
Range, is a momentum indicator used to spot entry and
exit points in the market by comparing the closing price
of the stock to the high-low range of prices in the look-
back window (W).
10) Disparity Index ∈ RN
+ : Its value is a percentage that
indicates the relative position of the current closing price
of the stock to a selected moving average. In this study,
the selected moving average is the EMA of the look-
back window (W).
C. Sentiment Scores
The supply and demand ﬂuctuations in the stock market are
highly sensitive to the moment’s news due to the impact of
mass media on the investor’s behavior. Hence many traders and
investors consider the news reports in their stock-picking strat-
egy. In our proposed approach, we believe that incorporating
the general news sentence towards the asset being considered
in the observation (state) deﬁnition will help the agent learn a
better trading strategy. In Ding et al. [31] study, they showed
that news headlines are more useful in forecasting than using
the entire news article content. Therefore, we only consider
news headlines as our input to calculate the sentiment score.
We describe the process of calculating a sentiment score for
each asset in the portfolio at time step t (day) as the following:
• We use a rule-based matching approach to search for
the asset name, stock symbol, or other keywords in the
headline news (ex. Microsoft or MSFT, tech,..) released
on day t.
• Then we use a ﬁne-tuned BERT model called FinBERT
[32] to calculate the sentiment probability (Positive,
Negative, or Neutral) of each news headline. FinBERT
model is a pre-trained NLP model to analyze sentiments
speciﬁcally for ﬁnancial text.
• Finally, we take the average of the asset’s news sentiment
probabilities for each day and assign 1 if the positive
probability is higher than the negative probability and -1
otherwise. We ignore the neutral probability as we believe
that if an asset has been mentioned on the news, it will
impact the asset price (positively or negatively). If the
asset has no news on a given day, we assign 0 to the
sentiment score.
V. EXPERIMENTS AND RESULTS
We evaluate our proposed approach by performing two
different back-testing which is the process used by traders and
analysts to assert the viability of a trading strategy by testing
it on historical data.
$$
We perform two different back-testing experiments,
the
purpose of the ﬁrst experiment (Section. V-B) is to validate the
superiority of the continuous action space to solve the trading
problem by comparing the results of the same experiments
reported by Kaur [39]. In their paper a discrete action space
is adapted to solve the problem, where the agent can choose to
buy, sell or hold action (i.e., discrete action space) of a ﬁxed
number of shares on each time step for a portfolio of two
assets, namely; Qualcomm (QCOM) and Microsoft (MSFT).
We back-test our approach on the same 5-years daily historical
stock data (between 2011-2016) used in their study with the
same amount of initial capital ($10,000).
$$
The second experiment (Section. V-C) is conducted to
validate the robustness of our model on large space of actions

and states by considering 10 different assets in the portfolio. In
addition, considering that the ﬁrst experiment was on training
data set only, we evaluate the performance on an unseen
market data (test data set) to check the agent’s ability of
generalization.
We use two metrics to evaluate our results: the ﬁrst metric is
the cumulative sum of reward, i.e., the total proﬁts at the end
of the trading episode. The second metric is the annualized
Sharpe ratio [38] that combines the return and the risk to give
the average of the risk-free return by the portfolio’s deviation.
In general, a Sharpe ratio above 1.0 is considered to be “good”
by investors because this suggests that the portfolio is offering
excess returns relative to its volatility. A Sharpe ratio higher
than 2.0 is rated as “very good” where a ratio above 3.0 is
considered “excellent”.
A. Data Description and Preprocessing
In this work, We use Yahoo Finance [33] to retrieve
historical market daily prices. The retrieved historical data
consists of 7 columns; Date, Volume, Open, Close, Adjusted
Close, High and Low prices. To prepare each dataset to be
used by the model, we ﬁrst perform timestamps processing by
using the trading calendar (exchange-calendars package [34])
to check if the market was open on the given dates to the
agent and exclude weekends and holidays from the dataset
so the agent will not face gaps in the trading process. Further
dataset processing is required to ensure that all ﬁnancial assets
(stocks) considered in the portfolio have an equal length of
historical data points. Some stocks have been recorded for
decades, while other newly listed stocks are only a few months.
This time-dimension alignment of stocks’ historical data will
prevent the bias action of the agent towards the stock with
more data. Once we have the timestamps processed we use
Close, High, Low prices and Volume at each timestamp to
calculate the technical indicators of each asset with a look-
back window (W).
To obtain a comprehensive and accurate ﬁnancial news,
we combined headline news from Benzinga, Seeking Alpha,
Zacks and other ﬁnancial news websites [35], and crawled
historical news headlines from Reddit worldNews Channel
[36]. The ﬁnal dataset consists of 3,288,724 news headlines
ranging between 2009-2021, which we utilized to calculate the
sentiment score.
B. First Experiment
In the ﬁrst experiment, we conduct three evaluations similar
to the benchmark paper [39]. All three evaluations share the
same conﬁgurations like the number of assets in the portfo-
lio, initial capital, commission rates, etc. but with different
components of the environment’s state representation. We start
with a baseline which only contains the close price as a
market signal feature, we then add technical indicators in
the second evaluation, and ﬁnally, we evaluate by adding
sentiment analysis scores. In Table. I we summarize the three
evaluations results of the experiment.
different factors such as the actions the agent randomly starts
with and uses to explore or the random weight initialization.
As suggested in [37] to ensure fairness and reliability of
our results, we average multiple runs over different random
seeds to have an insight
into the population distribution
of the algorithm performance on an environment. In this
results
experiment’s evaluations, we report and highlight
across several
independent runs. While the recommended
number of trials to evaluate an RL algorithm is still an open
question in the ﬁeld, we reported the mean and standard error
across ﬁve trials (runs), which is the suggested number in
many studies [37].
1) Evaluation on Baseline Environment: To evaluate the
continuous action approach in our model, we test it by solving
the problem with only the close price of the assets as a
market signal; hence the state representation in this baseline
environment consists of only the position state and the close
price of the asset at t (Ct) as a market signal, i.e., the agent
will solely make its trading decisions based on merely the
closing price of the stock as a market feature. We perform
5 experiment trials each with 200 epochs (episodes) for the
same hyperparameter conﬁguration, only varying the random
seed across trials.
$$
Fig. 2 shows the average return (sum of rewards) at each
trading episode and the standard error across the 5 runs. As
can be observed, the agent’s performance increases with more
experience it gains with the number of epochs to successfully
achieve 33960$ average return (proﬁts) with standard error
equals to ±4473$. From the commission spent by the agent,
we can conclude that the agent was successfully able to ﬁnd
a balanced trading strategy by balancing between trading and
holding positions. Finally, the average annual Sharpe ratio of
our approach on the baseline environment was 1.43 with a
standard error of ±0.13. This is signiﬁcantly higher than the
reported Sharpe ratio 0.85 in [39] benchmark.
$$
2) Evaluation on WithTechIndicators Environment: Using
the same conﬁgurations used in baseline environment evalua-
tion, we augment the state with technical indicators and run 5
independent experiments to report the average return, Sharpe
ratio, and commission. We refer to this environment with
technical indicators and close price in the state representation
as WithTechIndicators environment.
$$
The results in Fig. 3 demonstrate that augmenting the en-
vironment with technical indicators has brought more helpful
information to the agent to make better decisions. The agent
successfully achieved 89782$ average return (proﬁts) with
±18980$ standard error, and an average Sharpe ratio equals
2.75 with a standard error ±0.43. We can also notice that
the average amount of commission is almost two times the
amount spent in the baseline environment, which means that
the agent was signiﬁcantly more active in buying/selling stocks
and closed more successful deals. In addition, our approach
outperformed the benchmark reported Sharpe ratio of 1.4.
$$
the
Due to the stochasticity in the learning process,
experiment results may change at each run depending on
3) Evaluation on WithSentiments Environment: We refer
to this environment with sentiment analysis scores, technical

Fig. 2.
TD3 agent performance metrics on Baseline environment using the same hyperparameter conﬁgurations averaged over 5 different random seeds.
a) Average return (Proﬁts in dollars) at the end of each episode. b) The average annual Sharpe ratio at the end of each episode. c) The average amount of
commission spent at the end of each episode
Fig. 3. TD3 agent performance metrics on WithTechIndicators Environment
using the same hyperparameter conﬁgurations averaged over 5 different
random seeds. a) Average return (Proﬁts in dollars) at the end of each episode.
b) The average annual Sharpe ratio at the end of each episode. c) The average
amount of commission spent at the end of each episode
Fig. 4.
TD3 agent performance metrics on WithSentiments Environment
using the same hyperparameter conﬁgurations averaged over 5 different
random seeds. a) Average return (Proﬁts in dollars) at the end of each episode.
b) The average annual Sharpe ratio at the end of each episode. c) The average
amount of commission spent at the end of each episode
$$
indicators, and close price in the state representation as With-
Sentiments environment. We include the sentiment scores of
news headlines for each asset in the state representation and
repeat the experiment with the same conﬁgurations. The total
average return proﬁts increased to 115591$ with standard error
equals to ±17721 across the ﬁve runs. Sharpe ratio increased
to 3.14 and ±0.40 standard error. The average amount of
commission equals the amount spent in the environment with
only technical indicators (WithTechIndicators environment),
which means that the agent performed almost the same number
of trades but with a better decision (policy). In the bench-
mark [39] study, they also reported an increase in the agent
performance when adding sentiment scores to the state with
a Sharpe ratio equal to 2.4. The plot showing the results in
Fig. 4 demonstrates that augmenting the state with sentiment
analysis along with technical indicators has improved the agent
performance.
$$
Experiment’s Summary
We notice in all plots of the three evaluations that the policy
improves over time, as the agent accumulates more reward,
and thus the Sharp ratio increases. Towards the end the slope
is almost ﬂat indicating that the policy has stabilized to local
optimum. As the stock trading problem has never been solved
we do not have a speciﬁed reward or Sharpe ratio threshold
at which it’s considered solved.
TABLE I
THE PERFORMANCE EVALUATION COMPARISON BETWEEN
THREE DIFFERENT EVALUATIONS AND BENCHMARK
Evaluation Environment
Accumulated Return
Sharpe Ratio
Commission
Sharpe Ration
benchmark
$$
Baseline
33960$ ±4473
1.43 ±0.13
355$ ±83
$$
$$
WithTechIndicators WithSentiments
115591$ ±17721
3.14 ±0.4
1447$ ±268
$$
$$
89782$ ±18980
2.75 ±0.43
1109$ ±248
$$
0.85
1.4
2.4

C. Second Experiment
In the second experiment, we evaluate our approach on a
wider action and state spaces by considering 10 assets to trade,
AAPL, MSFT, QCOM, IBM, RTX, PG, GS, NKE, DIS and
AXP.
$$
Our back-testing uses historical daily data from 01/01/2010
to 01/01/2018 with initial capital of 100000$ for performance
evaluation. We split the data set into two periods, the ﬁrst
period is to train the agent, the second is used to test the
performance of the agent on unseen data (Fig. 5).
$$
Fig. 5. Train, and test data splits
$$
We notice that for our model to generalize better, we had
to impose regularization by normalizing the observation space
using Batch Normalization. This technique uses mini-batches
from samples to have unit mean and variance. It maintains a
running moving average of the mean and variance to normalize
the observation vector during testing. We further normalized
the rewards received by the agent as it makes the gradient
steeper for better rewards. We also set the look-back window to
20 (W = 20). We added action noise to encourage exploration
during training to force the agent to try different actions and
explore its environment more effectively, leading to higher
rewards and more elegant behaviors.
$$
$$
Our approach successfully archived a 2.68 Sharpe ratio
which considered “very good” and 110308$ as total proﬁts
(Rewards) on the test data. We let the agent keep learning on
the test set since this will help the agent better adapt to the
market dynamics.
$$
Disabling Sell Action: To investigate whether proﬁts made
on the test data (between 2016 and 2018) are a matter of
the standard increase in the stocks and the market growth
in general (primarily that tech stocks are known for their
excellent performance in the past years) or are made due to
the decision made by the agent, we disable the sell action and
only let the agent buy and hold during the trading episode. As
a result the agent allocated the capital as follows:
Stock
AAPL
AXP
MSFT
IBM
DIS
Shares
751
398
398
200
199
$$
and hold on to this position until the end of the episode.
The Sharpe ratio has decreased to 2.00 with 66949$ as total
proﬁts (Rewards), which indicates that the decision made by
the agent had a positive effect on the return and it was not
merely due to the natural growth of the market.
$$
VI. CONCLUSION AND FUTURE WORKS
This work presented a Deep Reinforcement Learning ap-
proach that combines technical indicators with sentiment anal-
ysis to ﬁnd an optimal trading policy for assets in the stock
market. Results show that the addition of technical indicators
and sentiment scores of the news headlines to the state repre-
sentation has signiﬁcantly improved the agent’s performance
and the superiority of using a continuous action space over a
discrete one to solve the trading problem. We also explored the
potential of using an Actor-Critic algorithm (TD3) to solve the
portfolio allocation problem. Our approach achieved an annual
Sharpe ratio of 2.68 on test data, which is considered ”Good”
by investors. The approach can be improved in future work
by having more computational power to run more experiences
and better evaluate the approach. Our environment, agent, and
learning process possess many hyperparameters that must be
tuned. It will be interesting to see the model’s performance
with better-tuned parameters, which requires high computation
power. In addition, we believe that training an NLP algorithm
to process the ﬁnancial news content
instead of only the
headline may positively affect the agent performance.
REFERENCES
[1] J. Patel, S. Shah, P. Thakkar, and K. Kotecha, “Predicting stock and
stock price index movement using trend deterministic data preparation and
machine learning techniques,” Expert Systems with Applications, vol. 42,
no. 1, pp. 259–268, 2015.
[2] A. Tsantekidis, N. Passalis, A. Tefas, J. Kanniainen, M. Gabbouj, and
A. Iosiﬁdis, “Forecasting stock prices from the limit order book using
convolutional neural networks,” in 2017 IEEE 19th Conference on Busi-
ness Informatics (CBI), vol. 01, pp. 7–12, 2017.
[3] A. Ntakaris, J. Kanniainen, M. Gabbouj, and A. Iosiﬁdis, “Mid-price pre-
diction based on machine learning methods with technical and quantitative
indicators,” PLOS ONE, vol. 15, pp. 1–39, 06 2020.
[4] Y. Hao and Q. Gao, “Predicting the trend of stock market index using
the hybrid neural network based on multiple time scale feature learning,”
Applied Sciences, vol. 10, no. 11, 2020.
[5] M. M. L. de Prado, “The 10 reasons most machine learning funds fail,”
WGSRN: Data Collection & Empirical Methods (Topic), 2018.
[6] T. L. Meng and M. Khushi, “Reinforcement learning in ﬁnancial markets,”
Data, vol. 4, no. 3, 2019.
[7] M. L. Puterman, Markov decision processes: discrete stochastic dynamic
programming. John Wiley & Sons, 2010.
[8] S. Chakraborty, “Capturing ﬁnancial markets to apply deep reinforcement
learning,” 2019.
[9] M. R. Vargas, C. E. M. dos Anjos, G. L. G. Bichara, and A. G. Evsukoff,
“Deep leaming for stock market prediction using technical indicators and
ﬁnancial news articles,” in 2018 International Joint Conference on Neural
Networks (IJCNN), pp. 1–8, 2018.
[10] M. Corazza and F. Bertoluzzo, “Q-learning-based ﬁnancial
trading
systems with applications,” Working Papers 2014:15, Department of
Economics, University of Venice ”Ca’ Foscari”, 2014.
[11] Z. Tan, C. Quek, and P. Y. Cheng, “Stock trading with cycles: A ﬁnancial
application of anﬁs and reinforcement learning,” Expert Systems with
Applications, vol. 38, no. 5, pp. 4741–4755, 2011.
[12] Y. Deng, F. Bao, Y. Kong, Z. Ren, and Q. Dai, “Deep direct rein-
forcement learning for ﬁnancial signal representation and trading,” IEEE
Transactions on Neural Networks and Learning Systems, vol. 28, no. 3,
pp. 653–664, 2017.
[13] O. Alagoz, H. Hsu, A. J. Schaefer, and M. S. Roberts, “Markov decision
processes: A tool for sequential decision making under uncertainty,”
Medical Decision Making, vol. 30, no. 4, p. 474–483, 2009.
[14] R. S. Sutton, F. Bach, and A. G. Barto, Reinforcement Learning: An
Introduction. MIT Press Ltd, 2018.
[15] R. E. Bellman, Dynamic programming. Princeton University Press,
2010.
[16] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wier-
stra, and M. Riedmiller, “Playing atari with deep reinforcement learning,”
2013.

[17] V. Mnih, K. Kavukcuoglu, D. Silver, A. Rusu, J. Veness, M. Bellemare,
A. Graves, M. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen,
C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra,
S. Legg, and D. Hassabis, “Human-level control through deep reinforce-
ment learning,” Nature, vol. 518, pp. 529–33, 02 2015.
[18] H. van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning
with double q-learning,” CoRR, vol. abs/1509.06461, 2015.
[19] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa,
D. Silver, and D. Wierstra, “Continuous control with deep reinforcement
learning,” 2019.
[20] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approx-
imation error in actor-critic methods,” CoRR, vol. abs/1802.09477, 2018.
[21] F. Bertoluzzo and M. Corazza, “Testing different reinforcement learn-
ing conﬁgurations for ﬁnancial trading: Introduction and applications,”
Procedia Economics and Finance, vol. 3, pp. 68–77, 2012. International
Conference Emerging Markets Queries in Finance and Business, Petru
Maior University of Tˆırgu-Mures, ROMANIA, October 24th - 27th, 2012.
[22] L. Conegundes and A. C. M. Pereira, “Beating the stock market with a
deep reinforcement learning day trading system,” in 2020 International
Joint Conference on Neural Networks (IJCNN), pp. 1–8, 2020.
[23] C. Kirkpatrick and J. R. Dahlquist, “Technical analysis: The complete
resource for ﬁnancial market technicians,” 2006.
[24] J. R. Nofsinger, “The impact of public information on investors,” Journal
of Banking & Finance, vol. 25, no. 7, pp. 1339–1366, 2001.
[25] Z. Xiong, X.-Y. Liu, S. Zhong, H. Yang, and A. Walid, “Practical deep
reinforcement learning approach for stock trading,” 2018.
[26] “Investopedia – slippage deﬁnition.” https://www.investopedia.com/
terms/s/slippage.asp. [Online; accessed 02-October-2021].
[27] Z. Jiang, D. Xu, and J. Liang, “A deep reinforcement learning framework
for the ﬁnancial portfolio management problem,” 2017.
[28] A. Akhmetzyanov, R. Yagfarov, S. Gafurov, M. Ostanin, and A. Klim-
chik, “Continuous control in deep reinforcement learning with direct
policy derivation from q network,” in Human Interaction, Emerging
Technologies and Future Applications II, (Cham), pp. 168–174, Springer
International Publishing, 2020.
[29] T. T.-L. Chong, W.-K. Ng, and V. K.-S. Liew, “Revisiting the per-
formance of macd and rsi oscillators,” Journal of Risk and Financial
Management, vol. 7, no. 1, pp. 1–12, 2014.
[30] J. Granville, Granville’s New Key to Stock Market Proﬁts. Papamoa
Press, 2018.
[31] X. Ding, Y. Zhang, T. Liu, and J. Duan, “Using structured events to
predict stock price movement: An empirical investigation,” in Proceedings
of the 2014 Conference on Empirical Methods in Natural Language
Processing (EMNLP), (Doha, Qatar), pp. 1415–1425, Association for
Computational Linguistics, Oct. 2014.
[32] D. Araci, “Finbert: Financial sentiment analysis with pre-trained lan-
guage models,” 2019.
[33] “Yahoo ﬁnance.” https://ﬁnance.yahoo.com/.
“exchange-calendars.”
[34] G.
Manoim,
exchange-calendars/.
https://pypi.org/project/
[35] “Kaggle – daily ﬁnancial news for 6000+ stocks.” https://www.kaggle.
com/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests.
[Online; accessed 15-November-2021].
[36] “Kaggle – sun, j. (2016, august). daily news for stock market predic-
[Online; accessed
tion.” https://www.kaggle.com/aaron7sun/stocknews.
15-November-2021].
[37] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger,
“Deep reinforcement learning that matters,” 2019.
[38] W. F. Sharpe, “The sharpe ratio,” The Journal of Portfolio Management,
vol. 21, no. 1, pp. 49–58, 1994.
[39] S. Kau, “Algorithmic trading using reinforcement learning augmented
with hidden markov model. working paper, stanford university.,” 2017.

## Как запускать обучение TD3 на нужных данных

Для обучения используй:

```sh
python train.py data/GOOG_2010-2024-06.csv --strategy=TD3 --window-size=47 --batch-size=32 --episode-count=50 --td3-timesteps=100000 --td3-noise-sigma=0.1 --td3-save-name=td3_model
```

- `data/GOOG_2010-2024-06.csv` — основной тренировочный датасет
- OOS для теста: `data/GOOG_2024-07_2025-04.csv`

**Не забудь:** если меняешь путь к файлу, указывай его первым аргументом!
