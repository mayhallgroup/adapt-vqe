
# ADAPT-VQE
This is the simulation code for performing ADAPT-VQE calculations. Method detailed in Nature Communications, 10, 3007, (2019):[article](https://www.nature.com/articles/s41467-019-10988-2?utm_source=other_website&utm_medium=display&utm_content=leaderboard&utm_campaign=JRCN_2_LW_X-moldailyfeed&error=cookies_not_supported&code=55be89d0-c3c3-4c68-aee9-7424ed20999f)
|
[preprint](https://arxiv.org/abs/1812.11173)


### Installation
1. Download
    
        git clone https://github.com/mayhallgroup/adapt-vqe.git
        cd adapt-vqe/

2. create virtual environment (optional)
         
        virtualenv -p python3 venv
        source venv/bin/activate

3. Install

        pip install .

4. run tests
    
        pytest test/*.py
