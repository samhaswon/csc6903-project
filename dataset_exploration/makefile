investigate_w_vs_wh: investigate_w_vs_wh.c
	gcc -O3 -march=native -flto -Wall -Wextra -pedantic -std=c11 investigate_w_vs_wh.c -o /tmp/investigate_w_vs_wh_c
run_investigation: investigate_w_vs_wh
	time /tmp/investigate_w_vs_wh_c ./ireland_data 4 -30 30 0.05
