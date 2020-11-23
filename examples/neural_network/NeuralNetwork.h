// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__NeuralNetwork__h__
#define __PUJ_ML__NeuralNetwork__h__

#include <limits>
#include <vector>

#include "Layer.h"

// -- Some fwd decls
template< class _TANN > class ClassificationTrainer;

/**
 */
template< class _TScl >
class NeuralNetwork
{
public:
  using Self    = NeuralNetwork;
  using TLayer  = Layer< _TScl >;

  using TScalar     = typename TLayer::TScalar;
  using TColVector  = typename TLayer::TColVector;
  using TRowVector  = typename TLayer::TRowVector;
  using TMatrix     = typename TLayer::TMatrix;
  using TActivation = typename TLayer::TActivation;

  using TLayers = std::vector< TLayer >;

public:
  NeuralNetwork( );
  NeuralNetwork( const Self& o );
  virtual ~NeuralNetwork( ) = default;
  Self& operator=( const Self& o );

  void add( unsigned int i, unsigned int o, const std::string& f );
  void add( unsigned int o, const std::string& f );
  void add( const TMatrix& w, const TColVector& b, const std::string& f );
  void add( const TLayer& l );
  void load_topology( std::istream& is );

  void set( unsigned int l, const TMatrix& w, const TColVector& b );

  unsigned int number_of_layers( ) const;
  TMatrix& weights( unsigned int l );
  const TMatrix& weights( unsigned int l ) const;
  TColVector& biases( unsigned int l );
  const TColVector& biases( unsigned int l ) const;
  TActivation* sigma( unsigned int l );
  const TActivation* sigma( unsigned int l ) const;

  void setNormalizationOffset( const TColVector& o );
  void setNormalizationScale( const TMatrix& s );

  void init( );

  TMatrix f( const TMatrix& x ) const;
  void f( std::vector< TMatrix >& a, std::vector< TMatrix >& z ) const;
  TMatrix t( const TMatrix& x ) const;

protected:
  TMatrix _d( const unsigned int& l, const TMatrix& z ) const;
  void _read_from( std::istream& i );
  void _copy_to( std::ostream& o ) const;

protected:
  TLayers m_L;
  TColVector m_NormalizationOffset;
  TMatrix    m_NormalizationScale;

public:
  ///!
  friend std::istream& operator>>( std::istream& i, Self& n )
    {
      n._read_from( i );
      return( i );
    }

  ///!
  friend std::ostream& operator<<( std::ostream& o, const Self& n )
    {
      n._copy_to( o );
      return( o );
    }

  friend class ClassificationTrainer< Self >;
};

#endif // __PUJ_ML__NeuralNetwork__h__

// eof - $RCSfile$
