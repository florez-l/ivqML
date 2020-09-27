// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__NeuralNetwork__h__
#define __PUJ_ML__NeuralNetwork__h__

#include <limits>
#include <vector>

#include "Layer.h"

/**
 */
template< class _TScalar >
class NeuralNetwork
{
public:
  using Self    = NeuralNetwork;
  using TScalar = _TScalar;
  using TLayer  = Layer< TScalar >;

  using TMatrix     = typename TLayer::TMatrix;
  using TRowVector  = typename TLayer::TRowVector;
  using TColVector  = typename TLayer::TColVector;
  using TActivation = typename TLayer::TActivation;

  using TLayers = std::vector< TLayer >;

public:
  NeuralNetwork(
    const TScalar& epsilon =
    std::numeric_limits< TScalar >::epsilon( ) * TScalar( 10 )
    );
  NeuralNetwork( const Self& other );
  virtual ~NeuralNetwork( ) = default;
  Self& operator=( const Self& other );

  void add( unsigned int i, unsigned int o, const TActivation& f );
  void add( unsigned int o, const TActivation& f );
  void add( const TMatrix& w, const TColVector& b, const TActivation& f );
  void add( const TLayer& l );

  void init( bool randomly = true );

  TMatrix operator()( const TMatrix& x ) const;

  TScalar cost(
    std::vector< TMatrix >& dw,
    std::vector< TColVector >& db,
    std::vector< TColVector >& a,
    std::vector< TColVector >& z,
    std::vector< TColVector >& d,
    const TMatrix& X, const TMatrix& Y
    );
  void train(
    const TMatrix& X, const TMatrix& Y,
    const TScalar& alpha,
    const TScalar& lambda = TScalar( 0 ),
    std::ostream* os = nullptr
    );

protected:
  void _ReadFrom( std::istream& i );
  void _CopyTo( std::ostream& o ) const;

protected:
  TScalar m_Epsilon;
  TLayers m_L;

public:
  ///!
  friend std::istream& operator>>( std::istream& i, Self& n )
    {
      n._ReadFrom( i );
      return( i );
    }

  ///!
  friend std::ostream& operator<<( std::ostream& o, const Self& n )
    {
      n._CopyTo( o );
      return( o );
    }
};

#endif // __PUJ_ML__NeuralNetwork__h__

// eof - $RCSfile$
