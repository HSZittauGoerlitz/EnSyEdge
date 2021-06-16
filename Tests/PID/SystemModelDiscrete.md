$$
\begin{aligned}
\frac{y}{u} &= \frac{b_2 z^2 + b_1 z + b_0}{a_3 z^3 + a_2 z^2 + a_1 z + a_0} \\
y \cdot (a_3 z^3 + a_2 z^2 + a_1 z + a_0 ) &= u \cdot (b_2 z^2 + b_1 z + b_0 ) \\
a_3 y_{k+3} + a_2 y_{k+2} + a_1 y_{k+1} + a_0 y_k &= b_2 u_{k+2} + b_1 u_{k+1} + b_0 u_k \\
a_3 y_k + a_2 y_{k-1} + a_1 y_{k-2} + a_0 y_{k-3} &= b_2 u_{k-1} + b_1 u_{k-2} + b_0 u_{k-3}
\end{aligned}
$$
$$
\mathbf{y_k = \frac{b_2}{a_3} u_{k-1} + \frac{b_1}{a_3} u_{k-2} + \frac{b_0}{a_3} u_{k-3}
      - \frac{a_2}{a_3} y_{k-1} - \frac{a_1}{a_3} y_{k-2} - \frac{a_0}{a_3} y_{k-3}}
$$