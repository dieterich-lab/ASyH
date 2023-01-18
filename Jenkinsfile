pipeline {
    agent any
    stages {
        stage('Prepare venv') {
            steps {
                sh '''
                    python3 -m venv venv
                    . ./venv/bin/activate
                    pip install --upgrade pip
                    pip install -e '.[tests]'
                '''
            }

        }
        stage('Linting') {
            steps {
                sh '''
                    . ./venv/bin/activate
                    flake8 ASyH --max-line-length 140
                '''
            }
        }
        stage('Testing') {
            steps {
                sh '''
                    . ./venv/bin/activate
                    export PYTHONPATH=$(pwd)
                    pytest --junitxml results.xml tests --cov=ASyH --cov-report xml
                '''
            }
        }
    }
    post {
        always {
            junit 'results.xml'
            cobertura coberturaReportFile: 'coverage.xml'
        }
        failure {
            emailext to: "asyh@dieterichlab.org",
            subject: "jenkins build:${currentBuild.currentResult}: ${env.JOB_NAME}",
            body: "${currentBuild.currentResult}: Job ${env.JOB_NAME}\nMore Info can be found here: ${env.BUILD_URL}"
        }
    }
}
