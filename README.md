# Fisher's linear discriminant for binary classification from scratch

Fisher's criterion set a linear combination that maximize betweenclass and minimize within class  variance by maximizing this expression:

$J(\textbf{w}) = \frac{\textbf{w}^T \textbf{S}_B \textbf{w}}{\textbf{w}^T \textbf{S}_w \textbf{w}}$,

where $S_B$ and $S_w$ are the between-class and within-class covariance matrices of input data. For the two-class problem, the weight vector is calculated as: 

$\textbf{W} = S^{-1}_{\textbf{W}} (\textbf{m}_m -\textbf{m}_b)$.

Where $\textbf{W}$ is a optimal vector that project the two classes maximizing betweenclass and minimizing within class variance and  $S^{−1}_W$ is the inverse of $S_W$, also  $m_b$ and $m_m$  are mean vectors of class $b$ and $m$ .

Images below shows an optimal vector that was gotten by fisher's criteron and a non optimal vector:

![FC](https://github.com/aleperalesg/Fisher-s-criterion/assets/120703609/6aa72956-b1c5-4363-9236-0c8cf68a2645)

Finally, discriminant function is given by

$g(x) = [\textbf{z} + \frac{1}{2} (m_n + m_b)]^T \textbf{w} $,

whose response is in the range $[−∞, ∞]$. Thus, the classification rule is

```math
\begin{align}
\hat{y} = 
\begin{cases}
    \text{m}	& \text{if}\;\;g(\mathbf{x})>0, \\
    \text{b} & \text{otherwise}. 
\end{cases}

\end{align}
```

Images below shows dataset distribution and the kernel density estimation of predictions of Fisher's linear discriminant:

![fc2](https://github.com/aleperalesg/Fisher-s-criterion/assets/120703609/bc60e533-922d-4664-97c0-6f1a0a3b7a10)
