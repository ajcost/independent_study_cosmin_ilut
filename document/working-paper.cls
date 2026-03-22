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
\definecolor{CustomColorBlue}{HTML}{0000FF}
\definecolor{CustomColorGreen}{rgb}{0,.4,0}
\definecolor{CustomColorRed}{rgb}{.3,0,.7}
\definecolor{palepink}{RGB}{180,60,110}
\definecolor{DustyLavender}{HTML}{2c1c45}      % sections, deeper purple
\definecolor{Charcoal}{HTML}{554e4e}           % dark neutral
\definecolor{CharcoalOlive}{HTML}{1a7668}      % green-ish, less gray than original
\definecolor{SandDune}{HTML}{cfd2b2}           % light accent (backgrounds)
\definecolor{DustyRose}{HTML}{C9688A}          % deeper pink for links/citations
\definecolor{Plum}{HTML}{5B2C6F}               % dark plum for theorems/propositions

% -----------------------------
% Hyperref
% -----------------------------
\RequirePackage[colorlinks,breaklinks,
    bookmarks=false,
    pdfstartview=Fit,
    pdfview=Fit,
    linkcolor=blue,
    urlcolor=CustomColorGreen,
    citecolor=DustyRose,
    hyperfootnotes=false
]{hyperref}
\RequirePackage[figure,table]{hypcap}
\urlstyle{rm}

% -----------------------------
% Cleveref — smart cross-references (\ref -> "Section 2", "Definition 1", etc.)
% -----------------------------
\RequirePackage[nameinlink,capitalize]{cleveref}
\AtBeginDocument{\let\ref\cref}

% -----------------------------
% Title, abstract, figure/table labels
% -----------------------------
\let\LaTeXtitle\title
\renewcommand{\title}[1]{\LaTeXtitle{\begin{minipage}{0.75\textwidth}\centering\huge\rmfamily\scshape #1\end{minipage}}}
\renewcommand{\abstractname}{Abstract}
\renewenvironment{abstract}
  {\noindent\begin{center}\bfseries\abstractname\end{center}\vspace{-1em}\small\begin{quote}}
  {\end{quote}\vspace{-1em}}
\renewcommand{\figurename}{\color{black}Figure}
\renewcommand{\tablename}{\color{black}Table}

% -----------------------------
% Section headings
% -----------------------------
\RequirePackage{titlesec}

% Format Section
\titleformat{\section}
  {\color{black}\raggedright\Large\bfseries}
  {\thesection}{1em}{}

% Format Subsection
\titleformat{\subsection}
  {\color{black}\raggedright\large\bfseries}
  {\thesubsection}{1em}{}

% Format Subsubsection
\titleformat{\subsubsection}
  {\color{black}\raggedright\normalsize\bfseries}
  {\thesubsubsection}{1em}{}

% Format Paragraph (Note the [runin] tag to keep it inline!)
\titleformat{\paragraph}[runin]
  {\color{black}\normalsize\bfseries}
  {\theparagraph}{1em}{}

% -----------------------------
% Theorem environments
% -----------------------------
\newtheoremstyle{purplestyle}{12pt}{12pt}{\itshape}{}{\bfseries\color{Plum}}{.}{.5em}{}
\newtheoremstyle{greenstyle}{12pt}{12pt}{\itshape}{}{\bfseries\color{CustomColorGreen}}{.}{.5em}{}

\theoremstyle{purplestyle}
\newtheorem{assumption}{Assumption}
\newtheorem{proposition}{Proposition}
\newtheorem{theorem}{Theorem}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{remark}{Remark}

\theoremstyle{greenstyle}
\newtheorem{definition}[theorem]{Definition}

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
\AtBeginDocument{\setstretch{1.1}}
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

\endinput