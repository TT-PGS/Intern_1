# Phân tích số liệu

Đối với từng giải thuật heuristics và meta heuristics: (theo cách tính trong bài báo)

- tính thời gian chạy trung bình của tất cả các lần chạy trong results, phân chia theo bộ (jobs, machines, splitmin), và toàn bộ results (theo như cách tính trong bài báo)
- tính lower bound (LB) trung bình của tất cả các lần chạy trong results, phân chia theo bộ (jobs, machines, splitmin), và toàn bộ results
- tính lower bound percentage (%LB) trung bình của tất cả các lần chạy trong results, phân chia theo bộ (jobs, machines, splitmin), và toàn bộ results (theo như cách tính trong bài báo)

Đối với giải thuật DQN:

TODO:

- tìm ra tập train tối ưu (thử các giá trị từ 9 -> 90 như bảng TABLE I trong bài báo)
- từ tập train tối ưu n vừa tìm được, train n sample ứng với từng giá trị job (10, 20, 30, 40, 50) (như bảng TABLE II trong bài báo)

Từ hai tập data trên, hợp thành bảng TABLE III như trong bài báo
