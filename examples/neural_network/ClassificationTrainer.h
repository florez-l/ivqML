// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__ClassificationTrainer__h__
#define __PUJ_ML__ClassificationTrainer__h__

#include <functional>
#include <limits>
#include "NeuralNetwork.h"

/**
 */
template< class _TANN >
class ClassificationTrainer
{
public:
  using Self = ClassificationTrainer;

  using TNeuralNetwork = _TANN;
  using TScalar        = typename TNeuralNetwork::TScalar;
  using TLayer         = typename TNeuralNetwork::TLayer;
  using TColVector     = typename TNeuralNetwork::TColVector;
  using TRowVector     = typename TNeuralNetwork::TRowVector;
  using TMatrix        = typename TNeuralNetwork::TMatrix;
  using TActivation    = typename TNeuralNetwork::TActivation;
  using TLayers        = typename TNeuralNetwork::TLayers;

  enum ENormalizationType
  {
    None = 0,
    Rescale,
    Standardization,
    Decorrelation
  };

public:
  ClassificationTrainer(
    TNeuralNetwork* ann,
    const TScalar& epsilon =
    std::numeric_limits< TScalar >::epsilon( )
    );
  ClassificationTrainer( const Self& other );
  virtual ~ClassificationTrainer( ) = default;
  Self& operator=( const Self& other );

  const TScalar& epsilon( ) const;
  const TScalar& learningRate( ) const;
  const TScalar& regularization( ) const;
  const unsigned long& batchSize( ) const;

  bool isDataNormalized( ) const;
  bool isDataRescaled( ) const;
  bool isDataStandardized( ) const;
  bool isDataDecorrelated( ) const;

  const TScalar& FtrainScore( ) const;
  const TScalar& FtestScore( ) const;
  const TScalar& FvalidScore( ) const;

  void setEpsilon( const TScalar& e );
  void setLearningRate( const TScalar& a );
  void setRegularization( const TScalar& l );
  void setBatchSize( const unsigned long& s );
  void setSizes( const double& train, const double& test );
  void setData( const TMatrix& X, const TMatrix& Y );

  void setNormalizationToNone( );
  void setNormalizationToRescale( );
  void setNormalizationToStandardization( );
  void setNormalizationToDecorrelation( );

  using TTrainObserver =
    std::function< void( const unsigned long&, const TScalar&, const TScalar& ) >;
  void train(
    const TTrainObserver& observer =
    []( const unsigned long& i, const TScalar& Jtrain, const TScalar& Jtest ){}
    );

protected:
  void _normalize( );
  void _train( const TTrainObserver& observer );
  TScalar _f1_score( const TMatrix& X, const TMatrix& Y );

protected:
  TScalar         m_Epsilon;
  TScalar         m_Alpha;
  TScalar         m_Lambda;
  unsigned long   m_BatchSize;
  TNeuralNetwork* m_Net;

  ENormalizationType m_NormalizationType;
  TColVector m_NormalizationOffset;
  TMatrix m_NormalizationScale;

  double m_Train;
  double m_Test;

  TMatrix m_X, m_Xtrain, m_Xtest, m_Xvalid;
  TMatrix m_Y, m_Ytrain, m_Ytest, m_Yvalid;
  TScalar m_FtrainScore, m_FtestScore, m_FvalidScore;
};

#endif // __PUJ_ML__ClassificationTrainer__h__

// eof - $RCSfile$
