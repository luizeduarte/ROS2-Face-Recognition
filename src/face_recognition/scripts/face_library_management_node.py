#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import joblib
from sklearn import svm

class face_library_management_node(Node):
    def __init__(self):
        super().__init__('face_library_management_node')

        # Carregar o modelo SVM e a biblioteca de embeddings
        self.model = joblib.load('svm_model.pkl')
        self.face_embeddings = np.load('face_embeddings.npy')
        self.face_labels = np.load('face_labels.npy')

	# assinatura para receber comandos de adição de rostos
        self.subscription_command = self.create_subscription(String, '/face_recognition/library_command', self.command_callback, 10)
	# assinatura para receber embeddings de rostos
        self.subscription_embedding = self.create_subscription(String, '/face_recognition/face_embedding', self.embedding_callback, 10)

        # Rótulo e Embedding temporários
        self.pending_label = None
        self.pending_embedding = None

        self.publisher_status = self.create_publisher(String, '/face_recognition/library_update_status', 10)
        
	

    def command_callback(self, msg):
        # Receber o comando para adicionar um novo rosto à biblioteca
        command_data = msg.data.split(';')
        if len(command_data) == 2 and command_data[0] == 'ADD_FACE':
            self.pending_label = command_data[1]  # Nome ou ID da pessoa
            self.get_logger().info(f'Received command to add new face with label: {self.pending_label}')
        else:
            self.get_logger().warning('Received unknown command.')

    def embedding_callback(self, msg):
        # Receber o embedding e adicionar à biblioteca se houver um rótulo pendente
        if self.pending_label:
            embedding_strings = msg.data.split(';')
            embeddings = [np.fromstring(embedding, sep=',') for embedding in embedding_strings if embedding]

            if len(embeddings) == 1:
                self.pending_embedding = embeddings[0]
                self.add_face_to_library(self.pending_embedding, self.pending_label)
                self.pending_label = None
                self.pending_embedding = None

    def add_face_to_library(self, embedding, label):
        # Adicionar o novo embedding e rótulo à biblioteca
        self.face_embeddings = np.vstack([self.face_embeddings, embedding])
        self.face_labels = np.append(self.face_labels, label)

        # Treinar novamente o modelo SVM com os novos dados
        self.model = svm.SVC(kernel='linear', probability=True)
        self.model.fit(self.face_embeddings, self.face_labels)

        # Salvar a biblioteca e o modelo atualizados
        joblib.dump(self.model, 'svm_model.pkl')
        np.save('face_embeddings.npy', self.face_embeddings)
        np.save('face_labels.npy', self.face_labels)

        # Publicar status de atualização
        status_msg = String()
        status_msg.data = f'Face with label {label} added to the library successfully.'
        self.publisher_status.publish(status_msg)

        self.get_logger().info(status_msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = face_library_management_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
