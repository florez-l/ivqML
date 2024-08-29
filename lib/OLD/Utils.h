// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivq__eigen__Utils__h__
#define __ivq__eigen__Utils__h__

#include <iostream>

#include <vector>
#include <random>
#include <set>

#include <ivq/eigen/Config.h>

namespace ivq
{
  namespace eigen
  {
    /**
     */
    template< class _S, class _T, class _X >
    void normalization_parameters(
      Eigen::EigenBase< _S >& S,
      Eigen::EigenBase< _T >& T,
      const Eigen::EigenBase< _X >& data
      )
    {
      using _R = double;
      using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;
      using _D = Eigen::DiagonalMatrix< typename _S::Scalar, Eigen::Dynamic >;

      auto X = data.derived( ).template cast< _R >( );
      auto mv = X.colwise( ).minCoeff( );
      auto Mv = X.colwise( ).maxCoeff( );
      _M d = Mv - mv;
      d = d.unaryExpr(
        []( _R v )
        {
          return( ( v != _R( 0 ) )? _R( 1 ) / v: _R( 1 ) );
        }
        );

      T.derived( ) = mv.template cast< typename _T::Scalar >( );
      S.derived( ) = _D( d.template cast< typename _S::Scalar >( ).row( 0 ) );
    }
 
    /**
     */
    template< class _S, class _T, class _X >
    void standardization_parameters(
      Eigen::EigenBase< _S >& S,
      Eigen::EigenBase< _T >& T,
      const Eigen::EigenBase< _X >& data
      )
    {
      using _R = double;
      using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;
      using _D = Eigen::DiagonalMatrix< typename _S::Scalar, Eigen::Dynamic >;

      auto X = data.derived( ).template cast< _R >( );
      auto m = X.colwise( ).mean( );
      auto C = X.rowwise( ) - m.row( 0 );
      _M O = ( C.transpose( ) * C ) / _R( X.rows( ) - 1 );
      _M d = O.diagonal( ).array( ).sqrt( ).unaryExpr(
        []( _R v )
        {
          return( ( v != _R( 0 ) )? _R( 1 ) / v: _R( 1 ) );
        }
        );

      T.derived( ) = m.template cast< typename _T::Scalar >( );
      S.derived( ) = _D( d.template cast< typename _S::Scalar >( ).col( 0 ) );
    }

   /**
     */
    template< class _Y, class _X >
    void categorize(
      std::vector< std::vector< Eigen::Index > >& C,
      Eigen::EigenBase< _Y >& Y,
      const Eigen::EigenBase< _X >& X
      )
    {
      using _S = typename _X::Scalar;
      struct _V1
      {
        void init( const _S& v, Eigen::Index i, Eigen::Index j )
          {
            this->operator()( v, i, j );
          }
        void operator()( const _S& v, Eigen::Index i, Eigen::Index j )
          {
            this->U.insert( v );
          }
        std::set< _S > U;
      } v1;
      X.derived( ).visit( v1 );

      Y.derived( ) = _Y::Zero( X.derived( ).rows( ), v1.U.size( ) );

      struct _V2
      {
        void init( const _S& v, Eigen::Index i, Eigen::Index j )
          {
            this->operator()( v, i, j );
          }
        void operator()( const _S& v, Eigen::Index i, Eigen::Index j )
          {
            Eigen::Index k =
              std::distance( this->U->begin( ), this->U->find( v ) );
            this->Y->operator()( i, k ) = 1;
            while( this->C->size( ) < k + 1 )
              this->C->push_back( std::vector< Eigen::Index >( ) );
            this->C->operator[]( k ).push_back( i );
          }
        _V2(
          std::vector< std::vector< Eigen::Index > >& c,
          _Y& y, const std::set< _S >& u
          )
          {
            this->C = &c;
            this->Y = &y;
            this->U = &u;
            this->C->clear( );
          }
        std::vector< std::vector< Eigen::Index > >* C;
        _Y* Y;
        const std::set< _S >* U;
      } v2( C, Y.derived( ), v1.U );
      X.derived( ).visit( v2 );
    }

    /**
     */
    template< class _Y, class _X >
    void balance(
      std::vector< Eigen::Index >& I,
      Eigen::EigenBase< _Y >& Y,
      const Eigen::EigenBase< _X >& X,
      double coeff = 1,
      bool categorize = false
      )
    {
      using _S = typename _Y::Scalar;
      if( categorize )
      {
        // Get categories
        std::vector< std::vector< Eigen::Index > > C;
        ivq::eigen::categorize( C, Y, X );

        // Shuffle data
        auto rd = std::random_device { };
        auto rng = std::default_random_engine { rd( ) };
        unsigned long long m =
          std::numeric_limits< unsigned long long >::max( );
        for( auto& c: C )
        {
          m = ( c.size( ) < m )? c.size( ): m;
          std::shuffle( c.begin( ), c.end( ), rng );
        } // end for
        m = ( unsigned long long )( double( m ) * coeff );

        // Fill results
        I.clear( );
        for( const auto& c: C )
          I.insert( I.end( ), c.begin( ), c.begin( ) + m );
        std::shuffle( I.begin( ), I.end( ), rng );
      }
      else
      {
        std::cerr << "ivq::eigen::balance(): Implement this." << std::endl;
        std::exit( 1 );
      } // end if
    }

    /**
     */
    template< class _K, class _Y, class _Z >
    void confusion(
      Eigen::EigenBase< _K >& K,
      const Eigen::EigenBase< _Y >& Y,
      const Eigen::EigenBase< _Z >& Z
      )
    {
      using _S = typename _K::Scalar;

      if( Y.derived( ).cols( ) == 1 )
      {
        _K mY( Y.derived( ).rows( ), 2 ), mZ( Z.derived( ).rows( ), 2 );
        mY << _S( 1 ) - Y.derived( ).array( ), Y.derived( );
        mZ << _S( 1 ) - Z.derived( ).array( ), Z.derived( );
        K.derived( ) = mY.transpose( ) * mZ;
      }
      else
        K.derived( ) = Y.derived( ).transpose( ) * Z.derived( );
    }

    /**
     */
    template< class _Y, class _C, class _M, class _X >
    void mahalanobis(
      Eigen::EigenBase< _Y >& Y,
      const Eigen::EigenBase< _C >& C,
      const Eigen::EigenBase< _M >& m,
      const Eigen::EigenBase< _X >& X
      )
    {
      auto D = X.rowwise( ) - m;
      Y = ( ( D * C.inverse( ) ) * D.transpose( ) ).diagonal( ).array( ).sqrt( );
    }

    /**
     */
    template< class _C1, class _C2, class _M1, class _M2 >
    typename _C1::Scalar bhattacharyya(
      const Eigen::EigenBase< _C1 >& P1,
      const Eigen::EigenBase< _C2 >& P2,
      const Eigen::EigenBase< _M1 >& m1,
      const Eigen::EigenBase< _M2 >& m2
      )
    {
      using _S = typename _C1::Scalar;
      using _X = Eigen::Matrix< _S, Eigen::Dynamic, Eigen::Dynamic >;

      _X P = ( P1.derived( ) + P2.derived( ) ).array( ) / _S( 2 );
      _X I = _X::Identity( P.rows( ), P.cols( ) );
      auto m = m1.derived( ) - m2.derived( );

      _S dP = P.determinant( );
      _S dP1 = P1.derived( ).determinant( );
      _S dP2 = P2.derived( ).determinant( );
      _S d = ( m.transpose( ) * P.llt( ).solve( I ) * m )( 0, 0 ) / _S( 8 );

      _S dP1P2 = dP1 * dP2;
      if( dP1P2 > _S( 0 ) && dP >= _S( 0 ) )
        d += std::log( ( dP + _S( 1e-8 ) ) / std::sqrt( dP1P2 ) ) / _S( 2 );

      return( d );
    }

    /**
     */
    template< class _X >
    Eigen::Matrix< unsigned long long, Eigen::Dynamic, Eigen::Dynamic >
    histogram( const Eigen::EigenBase< _X >& iX, unsigned short nBins = 100 )
    {
      using _R = double;
      using _C = Eigen::Matrix< _R, 1, Eigen::Dynamic >;
      using _H =
        Eigen::Matrix< unsigned long long, Eigen::Dynamic, Eigen::Dynamic >;

      auto X = iX.derived( ).template cast< _R >( );
      _C minX = X.colwise( ).minCoeff( );
      _C maxX = X.colwise( ).maxCoeff( );
      _C difX = ( maxX - minX ).array( )  / _R( nBins - 1 );
      auto Q = ( X.rowwise( ) - minX ).array( ).rowwise( ) / difX.array( );

      _H H = _H::Zero( nBins, X.cols( ) );
      for( unsigned long long r = 0; r < Q.rows( ); ++r )
        for( unsigned long long c = 0; c < Q.cols( ); ++c )
          H( ( unsigned long long )( Q( r, c ) ), c ) += 1;
      return( H );
    }

    /**
     */
    template< class _X >
    Eigen::Matrix< unsigned long long, Eigen::Dynamic, Eigen::Dynamic >
    cumulative_histogram(
      const Eigen::EigenBase< _X >& iX, unsigned short nBins = 100
      )
    {
      using _H =
        Eigen::Matrix< unsigned long long, Eigen::Dynamic, Eigen::Dynamic >;
      _H H = histogram( iX, nBins );
      for( unsigned long long r = 1; r < H.rows( ); ++r )
        H.row( r ) += H.row( r - 1 );
      return( H );
    }

    /**
     */
    template< class _X >
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >
    density( const Eigen::EigenBase< _X >& iX, unsigned short nBins = 100 )
    {
      using _R = double;
      using _C = Eigen::Matrix< _R, 1, Eigen::Dynamic >;
      using _H = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;

      auto X = iX.derived( ).template cast< _R >( );
      _C minX = X.colwise( ).minCoeff( );
      _C maxX = X.colwise( ).maxCoeff( );
      _C difX = ( maxX - minX ).array( )  / _R( nBins - 1 );
      auto Q = ( X.rowwise( ) - minX ).array( ).rowwise( ) / difX.array( );

      double o = double( 1 ) / double( X.rows( ) );
      _H H = _H::Zero( nBins, X.cols( ) );
      for( unsigned long long r = 0; r < Q.rows( ); ++r )
        for( unsigned long long c = 0; c < Q.cols( ); ++c )
          H( ( unsigned long long )( Q( r, c ) ), c ) += o;
      return( H );
    }

    /**
     */
    template< class _X >
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >
    cumulative_density(
      const Eigen::EigenBase< _X >& iX, unsigned short nBins = 100
      )
    {
      using _H = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >;
      _H H = density( iX, nBins );
      for( unsigned long long r = 1; r < H.rows( ); ++r )
        H.row( r ) += H.row( r - 1 );
      return( H );
    }

    /**
     */
    template< class _O, class _I >
    void equalize_histogram(
      Eigen::EigenBase< _O >& iO,
      const Eigen::EigenBase< _I >& iI,
      unsigned short nBins = 100
      )
    {
      using _R = double;
      using _C = Eigen::Matrix< _R, 1, Eigen::Dynamic >;

      auto O = iO.derived( );
      auto I = iI.derived( );

      _C minI = I.colwise( ).minCoeff( ).template cast< _R >( );
      _C maxI = I.colwise( ).maxCoeff( ).template cast< _R >( );
      _C difI = maxI - minI;
      auto cfd = cumulative_density( I, nBins );
      Eigen::Matrix< unsigned long long, Eigen::Dynamic, Eigen::Dynamic > X =
        (
          ( I.template cast< _R >( ).rowwise( ) - minI ).array( ).rowwise( )
          /
          difI.array( ) * _R( nBins - 1 )
          ).template cast< unsigned long long >( );
      for( unsigned long long c = 0; c < I.cols( ); ++c )
      {
        O.col( c ) =
          (
            (
              (
                (
                  ( cfd( X.array( ).col( c ), c ).array( )
                    *
                    _R( nBins - 1 )
                    ).ceil( ) - _R( 1 )
                  )
                /
                _R( nBins - 1 )
                ).rowwise( )
              *
              difI.array( )
              ).rowwise( ) + minI.array( )
            ).template cast< typename _O::Scalar >( );
      } // end for
    }
  } // end namespace
} // end namespace

#endif // __ivq__eigen__Utils__h__

// eof - $RCSfile$
