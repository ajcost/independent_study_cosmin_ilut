\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{working-paper}[2025/01/27 Article class for Working Papers]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

% -----------------------------
% Base class
% -----------------------------
\LoadClass[11pt]{article}

% -----------------------------
% Core packages
% -----------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{kpfonts}
\RequirePackage{amsmath,amssymb,amsfonts,amsthm}
\RequirePackage[margin=1.2in]{geometry}
\RequirePackage{graphicx}
\RequirePackage{natbib}
\RequirePackage{fancyhdr}
\RequirePackage{setspace}

% -----------------------------
% Colors
% -----------------------------
\RequirePackage{xcolor}
\definecolor{CustomColorDarkBlue}{rgb}{.1,0,.2}
\definecolor{CustomColorBlue}{rgb}{.1,.1,.5}
\definecolor{CustomColorGreen}{rgb}{0,.4,0}
\definecolor{CustomColorRed}{rgb}{.3,0,.7}
\definecolor{palepink}{RGB}{180,60,110}
\definecolor{DustyLavender}{HTML}{2c1c45}      % sections, deeper purple
\definecolor{Charcoal}{HTML}{554e4e}           % dark neutral
\definecolor{CharcoalOlive}{HTML}{1a7668}      % green-ish, less gray than original
\definecolor{SandDune}{HTML}{cfd2b2}           % light accent (backgrounds)
\definecolor{DustyRose}{HTML}{c94a7c}          % dark pink for links/citations

% -----------------------------
% Hyperref
% -----------------------------
\RequirePackage[colorlinks,breaklinks,
    bookmarks=false,
    pdfstartview=Fit,
    pdfview=Fit,
    linkcolor=CustomColorRed,
    urlcolor=CustomColorGreen,
    citecolor=DustyRose,
    hyperfootnotes=false
]{hyperref}
\RequirePackage[figure,table]{hypcap}
\urlstyle{rm}

% -----------------------------
% Title, abstract, figure/table labels
% -----------------------------
\let\LaTeXtitle\title
\renewcommand{\title}[1]{\LaTeXtitle{\begin{minipage}{0.75\textwidth}\centering\huge\rmfamily\scshape #1\end{minipage}}}
\renewcommand{\abstractname}{Abstract}
\renewenvironment{abstract}
  {\noindent\begin{center}\bfseries\abstractname\end{center}\vspace{-1em}\small\begin{quote}}
  {\end{quote}\vspace{-1em}}
\renewcommand{\figurename}{\color{DustyLavender}Figure}
\renewcommand{\tablename}{\color{DustyLavender}Table}

% -----------------------------
% Section headings
% -----------------------------
\let\LaTeX@startsection\@startsection
\renewcommand{\@startsection}[6]{\LaTeX@startsection%
{#1}{#2}{#3}{#4}{#5}{\color{DustyLavender}\raggedright #6}}

\renewcommand{\@seccntformat}[1]{%
  \csname the#1\endcsname\quad
}

% -----------------------------
% Theorem environments
% -----------------------------
\newtheoremstyle{assumptionstyle}{12pt}{12pt}{\itshape}{}{\bfseries\color{CharcoalOlive}}{.}{.5em}{}
\newtheoremstyle{propositionstyle}{12pt}{12pt}{\itshape}{}{\bfseries\color{CharcoalOlive}}{.}{.5em}{}
\newtheoremstyle{standardstyle}{12pt}{12pt}{\itshape}{}{\bfseries\color{CharcoalOlive}}{.}{.5em}{}

\theoremstyle{assumptionstyle}
\newtheorem{assumption}{Assumption}

\theoremstyle{propositionstyle}
\newtheorem{proposition}{Proposition}

\theoremstyle{standardstyle}
\newtheorem{theorem}{Theorem}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}{Remark}

\renewcommand{\qedsymbol}{\rule{0.5em}{0.5em}}

% -----------------------------
% Running heads
% -----------------------------
% -----------------------------
% Running heads
% -----------------------------
\pagestyle{fancy}
\fancyhf{}
\cfoot{\thepage}
\renewcommand{\headrulewidth}{0pt}

% Storage for author/title
\newcommand{\@shortauthor}{}
\newcommand{\@shorttitle}{}

\newcommand{\runningheads}[2]{%
  \renewcommand{\@shortauthor}{#1}%
  \renewcommand{\@shorttitle}{#2}%
  \fancyhead[CE]{\color{CharcoalOlive}\footnotesize\MakeUppercase{#1}}%
  \fancyhead[CO]{\color{CharcoalOlive}\footnotesize\MakeUppercase{#2}}%
}

% Make page 1 (plain style) also show headers
\fancypagestyle{plain}{%
  \fancyhf{}%
  \cfoot{\thepage}%
  \fancyhead[CE]{\color{CharcoalOlive}\footnotesize\MakeUppercase{\@shortauthor}}%
  \fancyhead[CO]{\color{CharcoalOlive}\footnotesize\MakeUppercase{\@shorttitle}}%
  \renewcommand{\headrulewidth}{0pt}%
}
% -----------------------------
% Spacing
% -----------------------------
\AtBeginDocument{\setstretch{1.125}}
\setlength{\parindent}{1.25em}
\setlength{\parskip}{0pt}

\setlength{\textfloatsep}{14pt plus 4pt minus 3pt}
\setlength{\floatsep}{12pt plus 4pt minus 3pt}
\setlength{\intextsep}{12pt plus 4pt minus 3pt}

% -----------------------------
% Helpers
% -----------------------------
\newcommand{\keywords}[1]{\noindent\textbf{Keywords:} #1\\}
\newcommand{\jelcodes}[1]{\noindent\textbf{JEL Codes:} #1\\}

\renewcommand{\maketitle}{%
  \begingroup
  \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
  \begin{center}%
    {\LARGE\scshape\@title\par}%
    \vskip 1em
    {\large\@author\par}%
    \vskip 0.5em
    {\@date\par}%
  \end{center}%
  \vskip 1.5em
  \endgroup
  \setcounter{footnote}{0}%
}

\endinput