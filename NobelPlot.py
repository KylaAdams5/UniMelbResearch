import matplotlib.pyplot as plt
import numpy as np

yearsMen = [1901,1902,
1903,
1904,
1905,
1906,
1907,
1908,
1909,
1910,
1911,
1912,
1913,
1914,
1915,
1916,
1917,
1918,
1919,
1920,
1921,
1922,
1923,
1924,
1925,
1926,
1927,
1928,
1929,
1930,
1931,
1932,
1933,
1934,
1935,
1936,
1937,
1938,
1939,
1940,
1941,
1942,
1943,
1944,
1945,
1946,
1947,
1948,
1949,
1950,
1951,
1952,
1953,
1954,
1955,
1956,
1957,
1958,
1959,
1960,
1961,
1962,
1963,
1964,
1965,
1966,
1967,
1968,
1969,
1970,
1971,
1972,
1973,
1974,
1975,
1976,
1977,
1978,
1979,
1980,
1981,
1982,
1983,
1984,
1985,
1986,
1987,
1988,
1989,
1990,
1991,
1992,
1993,
1994,
1995,
1996,
1997,
1998,
1999,
2000,
2001,
2002,
2003,
2004,
2005,
2006,
2007,
2008,
2009,
2010,
2011,
2012,
2013,
2014,
2015,
2016,
2017,2018]

men = [1,
3,
5,
6,
7,
8,
9,
10,
12,
13,
14,
15,
16,
17,
19,
19,
20,
21,
22,
23,
24,
25,
26,
27,
29,
30,
32,
33,
34,
35,
35,
36,
38,
38,
39,
41,
43,
44,
45,
45,
45,
45,
46,
47,
48,
49,
50,
51,
52,
53,
55,
57,
58,
60,
62,
65,
67,
70,
72,
73,
75,
76,
78,
81,
84,
85,
86,
87,
88,
90,
91,
94,
97,
99,
102,
104,
107,
110,
113,
115,
118,
119,
121,
123,
124,
127,
129,
132,
135,
138,
139,
140,
142,
144,
146,
149,
152,
155,
157,
160,
163,
166,
169,
172,
175,
177,
179,
182,
185,
187,
190,
192,
194,
197,
199,
202,
205,
207]

yearsWomen = [1903,1963,2018]
yearsFuture = [1903,1963,2018,2063,2090]
Women = [1,2,3]

fit = np.polyfit(yearsWomen,Women,2)
print(fit)
fit_fn = np.poly1d(fit)



# plt.plot(yearsMen,men, label = "Male Laureates ")
# plt.show()
# plt.hold()
plt.plot(yearsWomen, Women, 'b.',label = "Female Laureates ")
plt.plot(yearsFuture, fit_fn(yearsFuture), '-k', label = "projection")
plt.legend(loc = 'upper left')
plt.title("Nobel Physics Laureates")
plt.xlabel('Year')
plt.ylabel("Total Number of Laureates")
plt.show()


y = 0.0173*x-32
