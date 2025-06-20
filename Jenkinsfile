pipeline {
    agent any
    environment {
        REPO_PATH='.'
        APP_NAME='asyh'
    }
    stages {
        stage('Prepare venv') {
            steps {
                sh '''
                    ./venv/bin/python3 -m venv venv
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
                    mvn test
                '''
            }
        }
    }
    post {
        always {
            echo 'Wrapping up ...'

    junit(testResults: 'tests/*.xml', allowEmptyResults: true)
        }
        success {
            echo 'Build succeeded!'
        }
        failure {
            emailext to: "${ASYH_DEV_EMAILS}",
            subject: "jenkins build:${currentBuild.currentResult}: ${env.JOB_NAME}",
            body: "${currentBuild.currentResult}: Job ${env.JOB_NAME}\nMore Info can be found here: ${env.BUILD_URL}"
        }
}
}
