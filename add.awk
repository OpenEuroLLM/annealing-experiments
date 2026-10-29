#!/usr/bin/gawk -f

{
  sum += $1;
  n++;
}

END {
  printf("%s\ninputs: %s; average: %s\n", sum, n, sum / n);
} # END
