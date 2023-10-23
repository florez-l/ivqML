// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__h__
#define __ivqML__Model__Base__h__

#include <Eigen/Core>
#include <cmath>
#include <ostream>
#include <limits>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _S >
    class Base
    {
    public:
      using Self = Base;
      using TScalar = _S;
      using TNatural = unsigned long long;

      using TMatrix = Eigen::Matrix< _S, Eigen::Dynamic, Eigen::Dynamic >;
      using TMap = Eigen::Map< TMatrix >;
      using TConstMap = Eigen::Map< const TMatrix >;

    public:
      Base( const TNatural& n = 1 )
        {
          this->set_number_of_parameters( n );
        }
      virtual ~Base( )
        {
          if( this->m_T != nullptr )
            delete this->m_T;
        }

      template< class _O >
      Base( const _O& other )
        {
          this->set_number_of_parameters( other.m_P );
          for( TNatural i = 0; i < this->m_P; ++i )
            *( this->m_T + i ) = _S( *( other.m_T + i ) );
        }
      template< class _O >
      Self& operator=( const _O& other )
        {
          this->set_number_of_parameters( other.m_P );
          for( TNatural i = 0; i < this->m_P; ++i )
            *( this->m_T + i ) = _S( *( other.m_T + i ) );
          return( *this );
        }

      _S& operator[]( const TNatural& i )
        {
          static _S zero = 0;
          if( i < this->m_P )
            return( *( this->m_T + i ) );
          else
          {
            zero = 0;
            return( zero );
          } // end if
        }
      const _S& operator[]( const TNatural& i ) const
        {
          static const _S zero = 0;
          if( i < this->m_P )
            return( *( this->m_T + i ) );
          else
            return( zero );
        }

      template< class _D >
      Self& operator+=( const Eigen::EigenBase< _D >& d )
        {
          TMap( this->m_T, d.rows( ), d.cols( ) ) += d.derived( ).template cast< _S >( );
          return( *this );
        }
      template< class _D >
      Self& operator-=( const Eigen::EigenBase< _D >& d )
        {
          TMap( this->m_T, d.rows( ), d.cols( ) ) -= d.derived( ).template cast< _S >( );
          return( *this );
        }

      const TNatural& number_of_parameters( ) const
        {
          return( this->m_P );
        }
      void set_number_of_parameters( const TNatural& p )
        {
          if( this->m_P != p )
          {
            if( this->m_T != nullptr )
              delete this->m_T;
            this->m_T = ( p > 0 )? new _S[ p ]: nullptr;
            this->m_P = p;
          } // end if
          if( p > 0 )
            std::memset( this->m_T, 0, p * sizeof( _S ) );
        }

      _S* begin( )
        {
          return( this->m_T );
        }
      const _S* begin( ) const
        {
          return( this->m_T );
        }
      _S* end( )
        {
          return( this->m_T + this->m_P );
        }
      const _S* end( ) const
        {
          return( this->m_T + this->m_P );
        }

    protected:
      virtual void _to_stream( std::ostream& o ) const
        {
          o << this->m_P;
          for( TNatural i = 0; i < this->m_P; ++i )
            o << " " << *( this->m_T + i );
        }

    protected:
      _S* m_T { nullptr };
      TNatural m_P { 0 };

    public:
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._to_stream( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace


#include <Eigen/Dense>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _S >
    class Linear
      : public ivqML::Model::Base< _S >
    {
    public:
      using Self = Linear;
      using Superclass = ivqML::Model::Base< _S >;

      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;
      using TConstMap = typename Superclass::TConstMap;

    public:
      Linear( const TNatural& n = 0 )
        : Superclass( n + 1 )
        {
        }
      virtual ~Linear( ) = default;

      TNatural number_of_inputs( ) const
        {
          return( this->m_P - 1 );
        }
      void set_number_of_inputs( const TNatural& i )
        {
          this->set_number_of_parameters( i + 1 );
        }

      template< class _Y, class _X >
      void operator()(
        Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
        bool derivative = false
        ) const
        {
          using _YS = typename _Y::Scalar;
          using _YM = Eigen::Matrix< _YS, Eigen::Dynamic, Eigen::Dynamic >;

          auto X = iX.derived( ).template cast< _S >( );
          if( derivative )
            iY.derived( )
              << _YM::Ones( X.rows( ), 1 ), X.template cast< _YS >( );
          else
            iY.derived( ) =
              (
                ( X * TMap( this->m_T + 1, this->m_P - 1, 1 ) ).array( )
                +
                *( this->m_T )
                ).template cast< _YS >( );
        }
      
      template< class _Y, class _X >
      void fit(
        const Eigen::EigenBase< _X >& iX, const Eigen::EigenBase< _Y >& iY,
        const _S& l = 0
        )
        {
          auto X = iX.derived( ).template cast< _S >( );
          auto Y = iY.derived( ).template cast< _S >( );

          TNatural m = X.rows( );
          TNatural n = X.cols( );
          this->set_number_of_inputs( n );

          TMatrix R = TMatrix::Zero( this->m_P, this->m_P );
          R( 0, 0 ) = 1;
          R.block( 1, 1, n, n ) = ( X.transpose( ) * X ).array( ) / _S( m );
          R.block( 0, 1, 1, n ) = X.colwise( ).mean( );
          R.block( 1, 0, n, 1 ) = R.block( 0, 1, 1, n ).transpose( );

          if( l != _S( 0 ) )
          {
            /* TODO
               L = numpy.identity( n + 1 ) * l
               L[ 0 , 0 ] = 0
               R += L
            */
          } // end if

          TMatrix c = TMatrix::Zero( 1, this->m_P );
          c( 0, 0 ) = Y.mean( );
          c.block( 0, 1, 1, n ) =
            ( X.array( ).colwise( ) * Y.array( ).col( 0 ) )
            .colwise( ).mean( );
          TMap( this->m_T, 1, this->m_P ) = c * R.inverse( );
          

          /* TODO
             self.m_T = c @ numpy.linalg.inv( R )
          */
        }
    };
  } // end namespace
} // end namespace

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _S >
    class Logistic
      : public ivqML::Model::Linear< _S >
    {
    public:
      using Self = Logistic;
      using Superclass = ivqML::Model::Linear< _S >;

      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;
      using TConstMap = typename Superclass::TConstMap;

    public:
      Logistic( const TNatural& n = 0 )
        : Superclass( n )
        {
        }
      virtual ~Logistic( ) = default;

      template< class _Y, class _X >
      void operator()(
        Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
        bool derivative = false
        ) const
        {
          using _YS = typename _Y::Scalar;
          using _YM = Eigen::Matrix< _YS, Eigen::Dynamic, Eigen::Dynamic >;
          static const _YS _0 = _YS( 0 );
          static const _YS _1 = _YS( 1 );
          static const _YS _E = std::numeric_limits< _YS >::epsilon( );
          static const _YS _L = std::log( _1 - _E ) - std::log( _E );
          static const auto f = [&]( _YS z ) -> _YS
            {
              if     ( z >  _L ) return( _1 );
              else if( z < -_L ) return( _0 );
              else               return( _1 / ( _1 + std::exp( -z ) ) );
            };
          static const auto d = [&]( _YS z ) -> _YS
            {
              _YS s = f( z );
              return( s * ( _1 - s ) );
            };

          if( derivative )
          {
            _YM Z;
            this->Superclass::operator()( Z, iX, false );
            this->Superclass::operator()( iY, iX, true );
            Z = Z.unaryExpr( d );
            iY.derived( ).array( ).colwise( ) *= Z.col( 0 ).array( );
          }
          else
          {
            this->Superclass::operator()( iY, iX, false );
            iY.derived( ) = iY.derived( ).unaryExpr( f );
          } // end if
        }

      template< class _Y, class _X >
      void threshold(
        Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX
        ) const
        {
          using _YS = typename _Y::Scalar;
          using _YM = Eigen::Matrix< _YS, Eigen::Dynamic, Eigen::Dynamic >;
          static const _YS _0 = _YS( 0 );
          static const _YS _1 = _YS( 1 );
          static const _YS _T = _YS( 0.5 );
          static const auto t = [&]( _YS z ) -> _YS
            {
              return( ( z < _T )? _0: _1 );
            };
          this->operator()( iY, iX, false );
          iY.derived( ) = iY.derived( ).unaryExpr( t );
        }
    };
  } // end namespace
} // end namespace

#endif // __ivqML__Model__Base__h__

// eof - $RCSfile$
