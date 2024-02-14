# Évaluation

**Modalités d'évaluation du cours**

## Modalités

L'objectif général de l'évaluation de ce cours est de **mettre en pratique les notions étudiées** (bonnes pratiques de développement et mise en production) de manière appliquée et réaliste, i.e. à travers un projet basé sur une problématique "métier" et des données réelles. Pour cela, l'évaluation sera en deux parties :

1. **Par groupe de 3** : un projet à choisir parmi les 3 parcours (MLOps, app interactive / dashboard, publication reproductible + site web). Idéalement, on choisira un projet réel, effectué par exemple dans le cadre d'un cours précédent et qui génère un output propre à une mise en production.
2. **Seul** : effectuer une revue de code d'un autre projet. Compétence essentielle et souvent attendue d'un data scientist, la revue de code sera l'occasion de bien intégrer les bonnes pratiques de développement (cf. checklist ci-dessous) et de faire un retour bienveillant sur un autre projet que celui de son groupe.

> **⚠️ Avertissement**
>
> Ce projet doit mobiliser des données publiquement accessibles. La récupération et structuration de ces données peut faire partie des enjeux du projet mais celles-ci ne doivent pas provenir d'un projet antérieur de votre scolarité pour lequel le partage de données n'est pas possible.

# Checklist des bonnes pratiques de développement

Les bonnes pratiques de développement ci-dessous sont les **indispensables de ce cours**. Elles doivent être à la fois appliquées dans les **projets de groupe**, et à la base de la **revue de code individuelle**.

- [ ] **Utilisation de Git**
  - Présence d’un fichier `.gitignore` adapté au langage et avec des règles additionnelles pour respecter les bonnes pratiques de _versioning_
  - Travail collaboratif : utilisation de branches et des _pull requests_
- [ ] **Présence d’un fichier `README`** présentant le projet : contexte, objectif, comment l’utiliser ?
- [ ] **Présence d’un fichier `LICENSE`** déclarant la licence (_open-source_) d’exploitation du projet.
- [ ] **Versioning des packages** : présence d’un fichier `requirements.txt` ou d’un fichier d’environnement `environment.yml` pour conda

- [ ] **Qualité du code**
  - Respect des standards communautaires : utiliser un _linter_ et/ou un _formatter_
  - Modularité : un script principal qui appelle des modules

- [ ] **Structure des projets**
  - Respect des standards communautaires (_cookiecutter_)
  - Modularité du projet selon le modèle évoqué dans le cours:
    - Code sur GitHub
    - Données sur S3
    - Fichiers de configuration (_secrets_, etc.) à part

# Parcours MLOps

> ### Objectif
> 
> A partir d’un projet existant ou d’un projet type _contest Kaggle_, développer un modèle de ML répondant à une problématique métier, puis le déployer sur une infrastructure de production conformément aux principes du **MLOps**.

## Etapes :

- [ ] **Respecter la _checklist_ des bonnes pratiques de développement** ;
- [ ] **Développer un modèle de ML qui répond à un besoin métier** ;
- [ ] **Entraîner le modèle via validation croisée, avec une procédure de _fine-tuning_ des hyperparamètres** ;
- [ ] **Formaliser le processus de _fine-tuning_ de manière reproductible via MLFlow** ;
- [ ] **Construire une API avec Fastapi pour exposer le meilleur modèle** ;
- [ ] **Créer une image Docker pour mettre à disposition l'API** ;
- [ ] **Déployer l'API sur le SSP Cloud** ;
- [ ] **Industrialiser le déploiement en mode GitOps avec ArgoCD** ;
- [ ] **Gérer le monitoring de l'application : _logs, dashboard_ de suivi des performances, etc.**

